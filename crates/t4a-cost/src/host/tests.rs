use super::*;

const XEON: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 85\n\
model name\t: Intel(R) Xeon(R) Platinum 9242 CPU @ 2.30GHz\n\nprocessor\t: 1\n\
vendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 85\n";

#[test]
fn cpuinfo_gives_vendor_family_model_and_cpu_count() {
    let class = parse_cpuinfo(XEON).unwrap();
    assert_eq!(
        class,
        CpuClass {
            vendor: "GenuineIntel".into(),
            family: 6,
            model: 85,
            cpus: 2
        }
    );
    assert_eq!(parse_cpuinfo("processor : 0\n"), None);
}
