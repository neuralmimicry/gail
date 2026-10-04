# "Alexa, ask Aaron ..." skill

A custom Alexa skill whose backend is Gail itself. Alexa posts signed JSON
to the public endpoint below. Gail verifies the request, sends the question
through the normal orchestrated chat path (`gail-auto`) and speaks the
answer back.

```
https://gail.neuralmimicry.ai/v1/integrations/alexa
```

## How it works

Alexa cannot send a Gail bearer token, so the route is public and
authenticated with Amazon's request verification for self-hosted HTTPS
skills (`src/alexa.rs`):

1. **Certificate URL.** `SignatureCertChainUrl` must use `https`, have host
   `s3.amazonaws.com`, have a path starting with `/echo.api/` after `..`
   normalisation, and use port 443 if a port is given.
2. **Certificate chain.** Gail downloads the PEM chain (64 KiB limit, 3 s
   timeout, no redirects) and caches it for `cert_cache_ttl_seconds`. It is
   re-verified on every request: the chain must be in date, its SANs must
   include `echo-api.amazon.com`, and it must chain to a webpki (Mozilla) root.
3. **Signature.** `Signature-256` must be a valid RSA-SHA256 signature over
   the raw body. The legacy SHA-1 `Signature` header is only used when
   `Signature-256` is absent.
4. **Timestamp.** `request.timestamp` must be within 150 s of now.
5. **Skill id.** `session.application.applicationId` and
   `context.System.application.applicationId` must be in `alexa.skill_ids`.

Failures return 400 (malformed request, bad URL, stale timestamp, wrong
skill) or 401 (certificate or signature failure). While the skill is
disabled, the route returns 404.

Verified questions go through `GailService::complete`, the same function
that `/v1/chat/completions` uses for `model: gail-auto`. So Aria governance
(request and response checks), routing and the LLM ledger all apply. Traffic
is attributed to client id `alexa`, which is treated as holding the `llm`
scope. A short system prompt asks for plain spoken answers with no markdown,
and any markdown that slips through is removed before Gail produces the SSML
response. The last three exchanges are kept in Alexa `sessionAttributes`,
so follow-up questions have context.

### Timing

Alexa waits about 8 s. Gail gives the completion `answer_deadline_ms`
(default 6.5 s):

- If the answer is not ready after 1.5 s, Gail sends a Progressive Response
  ("One moment, let me think.") through `context.System.apiEndpoint` +
  `/v1/directives`. It does this only for `*.amazonalexa.com` endpoints.
- If the deadline passes, Alexa says "I'm still thinking about that one. Say
  yes to hear the answer ...". The completion keeps running for up to two
  minutes. Saying **yes** (`AMAZON.YesIntent`) delivers the answer, or
  waits another deadline for it.

## Invocation name

The invocation name is **`aaron`**, so you say "Alexa, ask Aaron what is
the tallest mountain in Scotland" or "Alexa, open Aaron".

Amazon's
[invocation-name rules](https://developer.amazon.com/en-US/docs/alexa/custom-skills/choose-the-invocation-name-for-a-custom-skill.html)
reject one-word names, and names of people unless they contain other
words. Both rules are enforced at **certification**. Developers report
that one-word names build and work in the development stage, and this
skill is never meant to be submitted.

If the console or `ask deploy` ever rejects the name, change
`invocationName` in both interaction models to `aaron assistant` and say
"Alexa, ask Aaron Assistant ...".

### Why there are several `AskAaron*` intents

The catch-all slot is `query` of type `AMAZON.SearchQuery`. Amazon requires
a carrier phrase before that slot, so a sample of just `{query}` fails the
build with `MissingCarrierPhraseWithPhraseSlot`. Alexa also strips the
carrier phrase from the slot value. To keep "what is X" from arriving as
"is X", the model splits questions by their first word: `AskAaronWhatIntent`
uses `what {query}`, `AskAaronWhoIntent` uses `who {query}`, and so on.
Gail puts the word back using `QUESTION_INTENTS` in `src/alexa.rs`.
`AskAaronIntent` is the catch-all for `about {query}`, `to {query}`,
`for {query}`, `i want to know {query}`, and similar phrases. A unit test
keeps the models and the table in step.

## Enable it in Gail

`gail.yaml`:

```yaml
alexa:
  enabled: true
  skill_ids: ["amzn1.ask.skill.xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"]
  # optional (defaults shown)
  timestamp_tolerance_seconds: 150   # clamped to 1..150
  answer_deadline_ms: 6500           # clamped to 500..7500
  cert_cache_ttl_seconds: 3600
  progressive_response: true
  max_answer_tokens: 350
```

The following environment variables override the file:

- `GAIL_ALEXA_ENABLED=true`
- `GAIL_ALEXA_SKILL_IDS=amzn1.ask.skill.a,amzn1.ask.skill.b` (comma- or
  space-separated)

Gail must be reachable at `https://gail.neuralmimicry.ai` with a publicly
trusted certificate. The current Let's Encrypt multi-SAN certificate is
fine, so `sslCertificateType` is `Trusted`. Its egress must also reach
`s3.amazonaws.com` and `api.*amazonalexa.com`.

## Deploy the skill (owner's Amazon developer account)

You need Node.js and the ASK CLI.

```bash
npm install -g ask-cli
ask configure                # sign in with the Amazon developer account
                             # that owns the Echo devices
cd integrations/alexa
ask deploy                   # creates the skill from skill-package/ and
                             # writes .ask/ask-states.json (do not commit it)
ask smapi get-skill-status -s <skill-id>   # wait for the build to succeed
```

`ask deploy` prints the new skill id (`amzn1.ask.skill....`). It is also in
`.ask/ask-states.json`. Put it in `alexa.skill_ids` (or
`GAIL_ALEXA_SKILL_IDS`), enable the route and redeploy Gail. Later model
changes are redeployed with `ask deploy` from the same directory.

## Use it on your Echo devices

Development-stage skills are enabled automatically for the developer
account. Every Echo registered to the **same Amazon account** can use the
skill straight away; there is no store listing and no certification. If a
device does not respond:

- check that testing is set to "Development" (`ask smapi set-skill-enablement
  -s <skill-id> -g development`, or the Test tab in the developer console);
- make sure the device's Alexa language is English (UK) or English (US).

## Test

```bash
# Interactive text session against the live endpoint
ask dialog --locale en-GB
#   User  > open aaron
#   User  > what is the tallest mountain in scotland
#   User  > stop

# One-shot simulation
ask smapi simulate-skill -s <skill-id> -l en-GB -g development \
  --input-content "ask aaron what is the tallest mountain in scotland"
ask smapi get-skill-simulation -s <skill-id> -i <simulation-id> -g development
```

Gail's own tests (`cargo test alexa`) cover certificate-URL rules, a
self-signed test chain, the real 2023 Amazon chain against webpki roots,
SHA-1 fallback, timestamp skew, skill-id mismatch, Launch/Intent/Stop
handling, the timeout fallback, Progressive Response, SSML escaping, and an
end-to-end signed request through the router with Aria in enforce mode.

## Files

- `ask-resources.json`: ASK CLI v2 project file.
- `skill-package/skill.json`: manifest (custom skill, en-GB and en-US,
  HTTPS endpoint, `Trusted` certificate, GB and US distribution, never
  submitted for certification).
- `skill-package/interactionModels/custom/en-GB.json` and `en-US.json`:
  interaction models.
