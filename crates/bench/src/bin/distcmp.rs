use std::fs;
use std::convert::TryInto;

fn quantize(v: &[f32]) -> Vec<i8> {
    v.iter().map(|&x| (x * 127.0).round().clamp(-127.0, 127.0) as i8).collect()
}
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| (*x as i32) * (*y as i32)).sum()
}
fn cos_dist_q(a: &[i8], b: &[i8]) -> f32 {
    let raw = dot_i8(a, b) as f32 / (127.0 * 127.0);
    1.0 - raw
}
fn f32_cos(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0;
    for i in 0..a.len() { d += a[i] * b[i]; }
    1.0 - d
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = fs::read(&path).unwrap();
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().add(8) as *const f32, (bytes.len()-8)/4).to_vec()
    };
    let q: Vec<Vec<i8>> = (0..n).map(|i| quantize(&data[i*dim..(i+1)*dim])).collect();
    // for 20 queries, compare top-10 by f32 vs int8
    let nq = 20.min(n);
    let mut overlap = 0u32;
    let mut total = 0u32;
    for qi in 0..nq {
        let qf = &data[qi*dim..(qi+1)*dim];
        let qi8 = &q[qi];
        let mut f32s: Vec<(usize,f32)> = (0..n).map(|i| (i, f32_cos(qf, &data[i*dim..(i+1)*dim]))).collect();
        let mut i8s: Vec<(usize,f32)> = (0..n).map(|i| (i, cos_dist_q(qi8, &q[i]))).collect();
        f32s.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        i8s.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let ft: Vec<usize> = f32s.iter().take(10).map(|(i,_)| *i).collect();
        let it: Vec<usize> = i8s.iter().take(10).map(|(i,_)| *i).collect();
        overlap += it.iter().filter(|x| ft.contains(x)).count() as u32;
        total += 10;
    }
    println!("int8 vs f32 top-10 overlap: {}/{} = {:.3}", overlap, total, overlap as f32/total as f32);
    // also check self-distance
    let s = cos_dist_q(&q[0], &q[0]);
    let sf = f32_cos(&data[0..dim], &data[0..dim]);
    println!("self dist q={:.4} f32={:.4}", s, sf);
}
