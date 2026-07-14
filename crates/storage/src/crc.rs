//! CRC32 (IEEE 802.3) — hand-rolled, table-based. Pure Rust, zero deps.
//! Used to checksum WAL records so torn writes are detected on recovery.

const POLY: u32 = 0xEDB88320;

struct Crc32Table([u32; 256]);

impl Crc32Table {
    const fn new() -> Self {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ POLY;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        Self(table)
    }
}

static TABLE: Crc32Table = Crc32Table::new();

/// Compute the CRC32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE.0[idx];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // "123456789" => 0xCBF43926 is the canonical CRC32 check value.
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7BE43);
    }
}
