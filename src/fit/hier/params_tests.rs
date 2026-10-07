use super::*;
use legume_numeric::candle::convert::to_host;

fn as_vec(v: &Var) -> Vec<f32> {
    to_host(v.as_tensor()).unwrap()
}

#[test]
fn init_is_seeded_and_biases_are_zero() {
    let dev = Device::Cpu;
    let a = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let b = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let c = HierParams::new(3, 2, 5, 4, 8, &dev).unwrap();
    assert_eq!(as_vec(&a.e_u), as_vec(&b.e_u));
    assert_ne!(as_vec(&a.e_u), as_vec(&c.e_u));
    assert_eq!(a.e_u.dims(), &[3, 4]);
    assert_eq!(a.mu.dims(), &[2, 4]);
    assert_eq!(a.r.dims(), &[5, 4]);
    assert!(to_host(a.b_m.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(to_host(a.b_g.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(as_vec(&a.e_u).iter().all(|x| x.abs() < 1.0));
}

/// A preset row composes back exactly (`μ_m + r_g = row`), the module mean is
/// the mean of its given rows, and the modes set what they pin.
#[test]
fn preset_rows_compose_back_exactly_and_the_mode_sets_the_pins() {
    let dev = Device::Cpu;
    let (h, module_of) = (2usize, vec![0u32, 0, 1, 1]);
    let given = PresetGenes {
        ids: vec![0, 1, 3],
        rows: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5],
        mode: PresetMode::Freeze,
    };
    let mut p = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    p.preset(&given, &module_of, &[]).unwrap();
    let (mu, r) = (as_vec(&p.mu), as_vec(&p.r));
    assert_eq!(
        &mu[0..2],
        &[2.0, 3.0],
        "module 0 mean of its two given rows"
    );
    assert_eq!(
        &mu[2..4],
        &[-1.0, 0.5],
        "module 1 mean of its one given row"
    );
    for (i, &g) in given.ids.iter().enumerate() {
        let m = module_of[g as usize] as usize;
        for k in 0..h {
            let composed: f32 = mu[m * h + k] + r[g as usize * h + k];
            assert!((composed - given.rows[i * h + k]).abs() < 1e-6);
        }
    }
    assert_eq!(
        p.mu_pinned,
        vec![true, true],
        "both modules have a given member"
    );
    assert!(p.is_frozen_gene(0) && !p.is_frozen_gene(2));
    let mask = to_host(p.r_mask.as_ref().unwrap()).unwrap();
    assert_eq!(mask, vec![0.0, 0.0, 1.0, 0.0]);
    assert!(p.lora.is_none());
    let (rho, _) = p.compose(&module_of).unwrap();
    for (i, &g) in given.ids.iter().enumerate() {
        for k in 0..h {
            assert_eq!(rho[(g as usize, k)], given.rows[i * h + k], "verbatim");
        }
    }

    let mut q = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    q.preset(
        &PresetGenes {
            mode: PresetMode::Init,
            ..given.clone()
        },
        &module_of,
        &[],
    )
    .unwrap();
    assert!(q.mu_pinned.is_empty() && q.r_mask.is_none() && q.frozen_gene.is_empty());

    let mut l = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    l.preset(
        &PresetGenes {
            mode: PresetMode::Lora(LoraSpec {
                rank: 1,
                lr_ratio: 4.0,
                ridge: 0.0,
            }),
            ..given.clone()
        },
        &module_of,
        &[],
    )
    .unwrap();
    let lora = l.lora.as_ref().expect("factors under lora");
    assert_eq!(lora.gene.u.dims(), &[4, 1]);
    assert_eq!(lora.gene.v.dims(), &[1, h]);
    assert_eq!(lora.module.u.dims(), &[2, 1]);
    assert_eq!(lora.module.v.dims(), &[1, h]);
    assert!(as_vec(&lora.module.u).iter().all(|&x| x != 0.0));
    assert!(as_vec(&lora.module.v).iter().all(|&x| x == 0.0));
    assert_eq!(
        to_host(&lora.gene.u_mask).unwrap(),
        vec![1.0, 1.0, 0.0, 1.0]
    );
    let u = as_vec(&lora.gene.u);
    assert!(u[0] != 0.0 && u[1] != 0.0 && u[2] == 0.0 && u[3] != 0.0);
    assert!(
        as_vec(&lora.gene.v).iter().all(|&x| x == 0.0),
        "the residual starts at nothing"
    );
    assert_eq!(l.mu_pinned, vec![true, true]);
    assert!(l.r_mask.is_some());
    assert!(HierParams::new(2, 2, 4, h, 1, &dev)
        .unwrap()
        .preset(
            &PresetGenes {
                mode: PresetMode::Lora(LoraSpec {
                    rank: h,
                    lr_ratio: 1.0,
                    ridge: 0.0
                }),
                ..given
            },
            &module_of,
            &[],
        )
        .is_err());
}

/// A free gene's composed row is its module's row plus its residual, and its
/// bias the module's plus its own.
#[test]
fn compose_adds_the_module_row_and_the_gene_residual() {
    let dev = Device::Cpu;
    let (h, module_of) = (3usize, vec![1u32, 0, 1]);
    let p = HierParams::new(2, 2, 3, h, 5, &dev).unwrap();
    p.b_m
        .set(&Tensor::from_vec(vec![0.5f32, -1.0], 2, &dev).unwrap())
        .unwrap();
    p.b_g
        .set(&Tensor::from_vec(vec![0.1f32, 0.2, 0.3], 3, &dev).unwrap())
        .unwrap();
    let (mu, r) = (as_vec(&p.mu), as_vec(&p.r));
    let (rho, b) = p.compose(&module_of).unwrap();
    assert_eq!(rho.shape(), (3, h));
    for (g, &m) in module_of.iter().enumerate() {
        let m = m as usize;
        for k in 0..h {
            assert!((rho[(g, k)] - (mu[m * h + k] + r[g * h + k])).abs() < 1e-6);
        }
    }
    let want = [-1.0 + 0.1, 0.5 + 0.2, -1.0 + 0.3];
    for (got, want) in b.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}
