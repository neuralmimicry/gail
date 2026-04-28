use serde::{Deserialize, Serialize};

use crate::errors::{GailError, Result};

pub const AER_MAGIC: &[u8; 4] = b"AER1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AerEvent {
    pub ts_us: u64,
    pub addr: u32,
    #[serde(default = "default_value")]
    pub value: u8,
}

const fn default_value() -> u8 {
    1
}

fn write_varint(value: u64, output: &mut Vec<u8>) {
    let mut current = value;
    while current >= 0x80 {
        output.push(((current as u8) & 0x7f) | 0x80);
        current >>= 7;
    }
    output.push((current as u8) & 0x7f);
}

fn read_varint(payload: &[u8], start: usize) -> Result<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0u32;
    let mut index = start;
    while index < payload.len() {
        let byte = payload[index];
        result |= ((byte & 0x7f) as u64) << shift;
        index += 1;
        if byte & 0x80 == 0 {
            return Ok((result, index));
        }
        shift += 7;
        if shift >= 64 {
            return Err(GailError::bad_request("AER varint overflow"));
        }
    }
    Err(GailError::bad_request("AER payload truncated"))
}

pub fn encode_events(events: &[AerEvent]) -> Vec<u8> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| event.ts_us);
    let base_ts = ordered[0].ts_us;
    let mut payload = Vec::with_capacity(ordered.len() * 8 + 12);
    payload.extend_from_slice(AER_MAGIC);
    payload.extend_from_slice(&base_ts.to_le_bytes());
    let mut previous_ts = base_ts;
    for event in ordered {
        let delta = event.ts_us.saturating_sub(previous_ts);
        previous_ts = event.ts_us;
        write_varint(delta, &mut payload);
        write_varint(u64::from(event.addr), &mut payload);
        write_varint(u64::from(event.value), &mut payload);
    }
    payload
}

pub fn decode_events(payload: &[u8]) -> Result<Vec<AerEvent>> {
    if payload.is_empty() {
        return Ok(Vec::new());
    }
    if payload.len() < 12 {
        return Err(GailError::bad_request("AER payload truncated"));
    }
    if &payload[..4] != AER_MAGIC {
        return Err(GailError::bad_request("AER payload magic mismatch"));
    }
    let mut ts_bytes = [0u8; 8];
    ts_bytes.copy_from_slice(&payload[4..12]);
    let base_ts = u64::from_le_bytes(ts_bytes);
    let mut previous_ts = base_ts;
    let mut index = 12usize;
    let mut events = Vec::new();
    while index < payload.len() {
        let (delta, next_index) = read_varint(payload, index)?;
        let (addr, next_index) = read_varint(payload, next_index)?;
        let (value, next_index) = read_varint(payload, next_index)?;
        previous_ts = previous_ts.saturating_add(delta);
        events.push(AerEvent {
            ts_us: previous_ts,
            addr: addr as u32,
            value: value as u8,
        });
        index = next_index;
    }
    Ok(events)
}

pub fn spikes_to_events(ts_us: u64, base_addr: u32, spikes: &[u8]) -> Vec<AerEvent> {
    spikes
        .iter()
        .enumerate()
        .filter_map(|(index, spike)| {
            if *spike == 0 {
                None
            } else {
                Some(AerEvent {
                    ts_us,
                    addr: base_addr + index as u32,
                    value: 1,
                })
            }
        })
        .collect()
}

pub fn encode_spikes(ts_us: u64, base_addr: u32, spikes: &[u8]) -> Vec<u8> {
    encode_events(&spikes_to_events(ts_us, base_addr, spikes))
}

pub fn apply_events_to_spikes(
    events: &[AerEvent],
    base_addr: u32,
    destination: &mut [u8],
) -> usize {
    let mut count = 0usize;
    for event in events {
        if event.value == 0 {
            continue;
        }
        let index = event.addr.saturating_sub(base_addr) as usize;
        if index < destination.len() {
            destination[index] = 1;
            count += 1;
        }
    }
    count
}

pub fn decode_spikes(payload: &[u8], base_addr: u32, length: usize) -> Result<Vec<u8>> {
    let events = decode_events(payload)?;
    let mut spikes = vec![0u8; length];
    apply_events_to_spikes(&events, base_addr, &mut spikes);
    Ok(spikes)
}

pub fn decode_spikes_auto(payload: &[u8], base_addr: u32) -> Result<Vec<u8>> {
    let events = decode_events(payload)?;
    let max_index = events
        .iter()
        .map(|event| event.addr.saturating_sub(base_addr) as usize)
        .max()
        .unwrap_or(0);
    let mut spikes = vec![0u8; max_index.saturating_add(1)];
    apply_events_to_spikes(&events, base_addr, &mut spikes);
    Ok(spikes)
}

pub fn spikes_from_floats(values: &[f32], threshold: f64) -> Vec<u8> {
    values
        .iter()
        .map(|value| if f64::from(*value) >= threshold { 1 } else { 0 })
        .collect()
}

pub fn payload_hex(payload: &[u8]) -> String {
    hex::encode(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aer_round_trip_preserves_events() {
        let payload = encode_events(&[
            AerEvent {
                ts_us: 100,
                addr: 4096,
                value: 1,
            },
            AerEvent {
                ts_us: 130,
                addr: 4099,
                value: 1,
            },
        ]);
        let decoded = decode_events(&payload).expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].addr, 4096);
        assert_eq!(decoded[1].ts_us, 130);
    }

    #[test]
    fn spike_vector_round_trip_preserves_indices() {
        let payload = encode_spikes(200, 4096, &[1, 0, 1, 0, 0, 1]);
        let spikes = decode_spikes(&payload, 4096, 6).expect("decode spikes");
        assert_eq!(spikes, vec![1, 0, 1, 0, 0, 1]);
    }
}
