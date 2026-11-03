use super::*;

#[test]
fn complex_layout_is_interleaved() {
    // The FFI/TAPP boundary and the packing code both assume this.
    assert_eq!(core::mem::size_of::<C64>(), 16);
    assert_eq!(core::mem::align_of::<C64>(), core::mem::align_of::<f64>());
    // `offset_of!` rather than a transmute to `[f64; 2]`: it asserts the same
    // thing without unsafe, and it asserts it more precisely — the field
    // *order* is named rather than inferred from which value came out first.
    assert_eq!(core::mem::offset_of!(C64, re), 0);
    assert_eq!(core::mem::offset_of!(C64, im), core::mem::size_of::<f64>());
}

#[test]
fn flops_per_mac() {
    assert_eq!(<f64 as Element>::FLOPS_PER_MAC, 2);
    assert_eq!(<C64 as Element>::FLOPS_PER_MAC, 8);
}
