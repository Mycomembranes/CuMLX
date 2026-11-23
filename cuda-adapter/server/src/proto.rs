// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framing and envelope helpers for `cuda-oxide.driver.v1`.
//!
//! Frame: `[u32 big-endian length][msgpack body]`, the same framing the rest of
//! the stack uses. Request body: `{op: str, seq: u32, args: map}`. Reply body:
//! `{status: "ok"|"error", seq, result, cu: i32, error: str}`.

use std::io::{Read, Write};

use rmpv::Value;

use crate::{cu, CuError, CuResult};

/// Largest frame accepted from a client. Bulk data above this goes in chunks.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
/// Largest single upload/download payload the forwarder puts in one frame.
pub const MAX_CHUNK_BYTES: usize = 16 * 1024 * 1024;

pub fn read_frame(stream: &mut impl Read) -> std::io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_be_bytes(len) as usize;
    if n > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame of {n} bytes exceeds the {MAX_FRAME_BYTES} byte cap"),
        ));
    }
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body)?;
    Ok(Some(body))
}

pub fn write_frame(stream: &mut impl Write, body: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(body.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "frame too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// A decoded request.
pub struct Request {
    pub op: String,
    pub seq: u64,
    pub args: Value,
}

pub fn decode_request(body: &[u8]) -> CuResult<Request> {
    let value = rmpv::decode::read_value(&mut &body[..])
        .map_err(|e| CuError::new(cu::INVALID_VALUE, format!("malformed msgpack request: {e}")))?;
    let op = get(&value, "op")
        .and_then(Value::as_str)
        .ok_or_else(|| CuError::new(cu::INVALID_VALUE, "request has no string `op`"))?
        .to_string();
    let seq = get(&value, "seq").and_then(Value::as_u64).unwrap_or(0);
    let args = get(&value, "args")
        .cloned()
        .unwrap_or(Value::Map(Vec::new()));
    Ok(Request { op, seq, args })
}

pub fn encode_ok(seq: u64, result: Value) -> Vec<u8> {
    encode(Value::Map(vec![
        (Value::from("status"), Value::from("ok")),
        (Value::from("seq"), Value::from(seq)),
        (Value::from("cu"), Value::from(cu::SUCCESS)),
        (Value::from("result"), result),
    ]))
}

pub fn encode_err(seq: u64, error: &CuError) -> Vec<u8> {
    encode(Value::Map(vec![
        (Value::from("status"), Value::from("error")),
        (Value::from("seq"), Value::from(seq)),
        (Value::from("cu"), Value::from(error.code)),
        (Value::from("error"), Value::from(error.reason.as_str())),
    ]))
}

fn encode(value: Value) -> Vec<u8> {
    let mut out = Vec::new();
    // Writing to a Vec cannot fail.
    rmpv::encode::write_value(&mut out, &value).expect("encode to Vec");
    out
}

pub fn get<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    map.as_map()?
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

pub fn arg_u64(args: &Value, key: &str) -> CuResult<u64> {
    get(args, key).and_then(Value::as_u64).ok_or_else(|| {
        CuError::new(
            cu::INVALID_VALUE,
            format!("missing integer argument `{key}`"),
        )
    })
}

pub fn arg_u32(args: &Value, key: &str) -> CuResult<u32> {
    u32::try_from(arg_u64(args, key)?).map_err(|_| {
        CuError::new(
            cu::INVALID_VALUE,
            format!("argument `{key}` exceeds 32 bits"),
        )
    })
}

pub fn arg_i32(args: &Value, key: &str) -> CuResult<i32> {
    get(args, key)
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("missing 32-bit argument `{key}`"),
            )
        })
}

pub fn arg_str<'a>(args: &'a Value, key: &str) -> CuResult<&'a str> {
    get(args, key).and_then(Value::as_str).ok_or_else(|| {
        CuError::new(
            cu::INVALID_VALUE,
            format!("missing string argument `{key}`"),
        )
    })
}

pub fn arg_bin<'a>(args: &'a Value, key: &str) -> CuResult<&'a [u8]> {
    match get(args, key) {
        Some(Value::Binary(bytes)) => Ok(bytes),
        _ => Err(CuError::new(
            cu::INVALID_VALUE,
            format!("missing binary argument `{key}`"),
        )),
    }
}

pub fn arg_dims(args: &Value, key: &str) -> CuResult<[u32; 3]> {
    let items = get(args, key)
        .and_then(Value::as_array)
        .filter(|items| items.len() == 3)
        .ok_or_else(|| {
            CuError::new(cu::INVALID_VALUE, format!("`{key}` must be three integers"))
        })?;
    let mut out = [0u32; 3];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                CuError::new(cu::INVALID_VALUE, format!("`{key}` entry is not a u32"))
            })?;
    }
    Ok(out)
}

pub fn map(pairs: Vec<(&str, Value)>) -> Value {
    Value::Map(
        pairs
            .into_iter()
            .map(|(k, v)| (Value::from(k), v))
            .collect(),
    )
}
