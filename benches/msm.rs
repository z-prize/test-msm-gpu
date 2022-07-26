// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_377::G1Affine;
use ark_ff::BigInteger256;

use std::str::FromStr;

use blst_msm::*;

extern "C" {
    fn hello();
    fn bc_get_device(dev: *mut ::std::os::raw::c_int) -> u32;
}

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("26".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    let mut val: ::std::os::raw::c_int = 5;

    unsafe {
        hello();
        bc_get_device(&mut val);
    }

    println!("{}", val);

    // let batches = 4;
    // let (points, scalars) =
    //     util::generate_points_scalars::<G1Affine>(1usize << npoints_npow, batches);
    // let mut context = multi_scalar_mult_init(points.as_slice());

    // let mut group = c.benchmark_group("CUDA");
    // group.sample_size(10);

    // let name = format!("2**{}x{}", npoints_npow, batches);
    // group.bench_function(name, |b| {
    //     b.iter(|| {
    //         let _ = multi_scalar_mult(&mut context, &points.as_slice(), unsafe {
    //             std::mem::transmute::<&[_], &[BigInteger256]>(
    //                 scalars.as_slice(),
    //             )
    //         });
    //     })
    // });

    // group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
