//! Deterministic LLM prompt budgeting.
//!
//! Provider context-window failures are deterministic and should be prevented
//! before an HTTP request consumes a queue slot.  This module keeps that policy
//! independent from provider transports so direct and orchestrated requests use
//! exactly the same compaction workflow.
//!
//! Compaction deliberately avoids a second LLM call: system instructions are
//! retained, the newest conversational turns are kept, and an explicit summary
//! records how much older history was omitted.  A single oversized message is
//! shortened in the middle so both its objective and most recent data survive.

use std::collections::BTreeMap;

use crate::models::{ChatMessage, ContentPart, MessageContent, ProviderCompletionRequest};

const MIN_CONTEXT_WINDOW_TOKENS: usize = 1_024;
const MIN_INPUT_BUDGET_TOKENS: usize = 256;
const COMPACTION_MARKER_BUDGET_CHARS: usize = 512;

/// Observable details for a request that was changed to fit a context window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptCompactionReport {
    pub context_window_tokens: usize,
    pub input_budget_tokens: usize,
    pub estimated_tokens_before: usize,
    pub estimated_tokens_after: usize,
    pub omitted_messages: usize,
    pub omitted_chars: usize,
}

/// Compact a provider request in place when its estimated input would exceed
/// the configured context window.
///
/// `safety_margin_tokens` covers chat templates and provider-specific tokeniser
/// variance. `max_tokens` is also reserved, so generated output cannot push the
/// complete request beyond the model context window.
pub fn compact_provider_request(
    request: &mut ProviderCompletionRequest,
    context_window_tokens: usize,
    chars_per_token: usize,
    safety_margin_tokens: usize,
) -> Option<PromptCompactionReport> {
    let context_window_tokens = context_window_tokens.max(MIN_CONTEXT_WINDOW_TOKENS);
    let chars_per_token = chars_per_token.max(1);
    let requested_output_tokens = request.max_tokens.unwrap_or(512) as usize;
    let effective_safety_margin = safety_margin_tokens.min(context_window_tokens / 4);
    let maximum_output_tokens = context_window_tokens
        .saturating_sub(effective_safety_margin)
        .saturating_sub(MIN_INPUT_BUDGET_TOKENS)
        .max(1);
    let output_tokens = requested_output_tokens.min(maximum_output_tokens);
    if requested_output_tokens > maximum_output_tokens {
        request.max_tokens = Some(maximum_output_tokens.min(u32::MAX as usize) as u32);
    }
    let input_budget_tokens = context_window_tokens
        .saturating_sub(output_tokens)
        .saturating_sub(effective_safety_margin)
        .max(MIN_INPUT_BUDGET_TOKENS);
    let input_budget_chars = input_budget_tokens.saturating_mul(chars_per_token);
    let before_chars = request_char_count(request);
    let estimated_tokens_before = estimate_tokens(before_chars, chars_per_token);
    if before_chars <= input_budget_chars {
        return None;
    }

    // System policy is more important than old conversation history. Give it a
    // bounded share so a pathological system prompt cannot evict the user task.
    let system_budget = input_budget_chars / 3;
    if let Some(system) = request.system.as_mut()
        && char_count(system) > system_budget
    {
        *system = truncate_middle(system, system_budget);
    }

    let retained_system_chars = request.system.as_deref().map(char_count).unwrap_or(0);
    let message_budget = input_budget_chars
        .saturating_sub(retained_system_chars)
        .saturating_sub(COMPACTION_MARKER_BUDGET_CHARS);

    let mut retained_reversed = Vec::new();
    let mut retained_chars = 0usize;
    let mut omitted_messages = Vec::new();
    for message in request.messages.iter().rev() {
        let message_chars = message_char_count(message);
        let remaining = message_budget.saturating_sub(retained_chars);
        if message_chars <= remaining {
            retained_reversed.push(message.clone());
            retained_chars = retained_chars.saturating_add(message_chars);
        } else if retained_reversed.is_empty() && remaining > 0 {
            retained_reversed.push(compact_message(message, remaining));
            retained_chars = message_budget;
        } else {
            omitted_messages.push(message);
        }
    }
    retained_reversed.reverse();

    let omitted_chars = omitted_messages
        .iter()
        .map(|message| message_char_count(message))
        .sum::<usize>();
    let omitted_message_count = omitted_messages.len();
    if !omitted_messages.is_empty() {
        let marker = compaction_summary(&omitted_messages, omitted_chars, chars_per_token);
        retained_reversed.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(truncate_middle(
                    &marker,
                    COMPACTION_MARKER_BUDGET_CHARS,
                )),
            },
        );
    }
    drop(omitted_messages);
    request.messages = retained_reversed;

    // The marker and UTF-8-safe truncation can be slightly conservative. Apply
    // one final cap to the newest text message if required.
    let after_first_pass = request_char_count(request);
    if after_first_pass > input_budget_chars
        && let Some(last) = request.messages.last_mut()
    {
        let overflow = after_first_pass - input_budget_chars;
        let target = message_char_count(last).saturating_sub(overflow);
        *last = compact_message(last, target);
    }

    let after_chars = request_char_count(request);
    Some(PromptCompactionReport {
        context_window_tokens,
        input_budget_tokens,
        estimated_tokens_before,
        estimated_tokens_after: estimate_tokens(after_chars, chars_per_token),
        omitted_messages: omitted_message_count,
        omitted_chars,
    })
}

fn request_char_count(request: &ProviderCompletionRequest) -> usize {
    request.system.as_deref().map(char_count).unwrap_or(0)
        + request
            .messages
            .iter()
            .map(message_char_count)
            .sum::<usize>()
}

fn message_char_count(message: &ChatMessage) -> usize {
    char_count(&message.flattened_text())
}

fn compact_message(message: &ChatMessage, max_chars: usize) -> ChatMessage {
    let content = match &message.content {
        MessageContent::Text(text) => MessageContent::Text(truncate_middle(text, max_chars)),
        MessageContent::Parts(parts) => {
            let mut remaining = max_chars;
            let compacted = parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => {
                        let compacted = truncate_middle(text, remaining);
                        remaining = remaining.saturating_sub(char_count(&compacted));
                        ContentPart::Text { text: compacted }
                    }
                    // Image payloads are not converted or discarded here. Their
                    // token accounting is provider-specific and they may be the
                    // primary input to a vision request.
                    ContentPart::ImageUrl { image_url } => ContentPart::ImageUrl {
                        image_url: image_url.clone(),
                    },
                })
                .collect();
            MessageContent::Parts(compacted)
        }
    };
    ChatMessage {
        role: message.role.clone(),
        content,
    }
}

fn compaction_summary(
    omitted: &[&ChatMessage],
    omitted_chars: usize,
    chars_per_token: usize,
) -> String {
    let mut roles = BTreeMap::<String, usize>::new();
    for message in omitted {
        *roles.entry(message.role.to_ascii_lowercase()).or_default() += 1;
    }
    let roles = roles
        .into_iter()
        .map(|(role, count)| format!("{role}:{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[GAIL CONTEXT COMPACTION] {} older message(s) (~{} tokens) were omitted to fit the provider context window. Omitted roles: {}. System instructions and the newest context are retained; re-query source data when omitted detail is required.",
        omitted.len(),
        estimate_tokens(omitted_chars, chars_per_token),
        roles,
    )
}

fn estimate_tokens(chars: usize, chars_per_token: usize) -> usize {
    chars.div_ceil(chars_per_token.max(1))
}

fn char_count(value: &str) -> usize {
    value.chars().count()
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    let length = char_count(value);
    if length <= max_chars {
        return value.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    const MARKER: &str = "\n…[middle omitted by Gail]…\n";
    let marker_chars = char_count(MARKER);
    if max_chars <= marker_chars + 2 {
        return value.chars().take(max_chars).collect();
    }
    let content_budget = max_chars - marker_chars;
    let head_chars = content_budget / 2;
    let tail_chars = content_budget - head_chars;
    let head = value.chars().take(head_chars).collect::<String>();
    let tail = value
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head}{MARKER}{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<ChatMessage>) -> ProviderCompletionRequest {
        ProviderCompletionRequest {
            provider: "ollama".to_string(),
            model: Some("qwen".to_string()),
            api_key: None,
            access_token: None,
            base_url: None,
            messages,
            system: Some("follow the policy".to_string()),
            max_tokens: Some(128),
            temperature: None,
            timeout_seconds: None,
            reasoning_effort: None,
            request_category: None,
            workflow: None,
            role: None,
            min_model_size_b: None,
            strict_no_downgrade: None,
        }
    }

    fn message(role: &str, value: impl Into<String>) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: MessageContent::Text(value.into()),
        }
    }

    #[test]
    fn leaves_requests_within_budget_unchanged() {
        let mut request = request(vec![message("user", "short prompt")]);
        let original = request.messages[0].flattened_text();
        assert!(compact_provider_request(&mut request, 16_384, 4, 1_024).is_none());
        assert_eq!(request.messages[0].flattened_text(), original);
    }

    #[test]
    fn retains_newest_turns_and_summarises_omitted_history() {
        let mut request = request(vec![
            message("user", "old-user".repeat(900)),
            message("assistant", "old-assistant".repeat(900)),
            message("user", "LATEST OBJECTIVE"),
        ]);
        let report = compact_provider_request(&mut request, 1_024, 2, 256)
            .expect("oversized request should be compacted");
        let combined = request
            .messages
            .iter()
            .map(ChatMessage::flattened_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(combined.contains("GAIL CONTEXT COMPACTION"));
        assert!(combined.contains("LATEST OBJECTIVE"));
        assert!(report.omitted_messages >= 1);
        assert!(report.estimated_tokens_after <= report.input_budget_tokens + 1);
    }

    #[test]
    fn oversized_single_message_keeps_its_head_and_tail() {
        let content = format!("START{}END", "x".repeat(10_000));
        let mut request = request(vec![message("user", content)]);
        compact_provider_request(&mut request, 1_024, 2, 256)
            .expect("oversized request should be compacted");
        let compacted = request.messages.last().unwrap().flattened_text();
        assert!(compacted.starts_with("START"));
        assert!(compacted.ends_with("END"));
        assert!(compacted.contains("middle omitted by Gail"));
    }
}
