//! The CPU class a provider's offline-fitted rows are keyed by.
//!
//! The class is the vendor, family and model of the first CPU and the number of
//! logical CPUs, read from `/proc/cpuinfo` (Linux): two SKUs of one
//! microarchitecture with different core counts are different classes, so a
//! provider's rows stay honest about which machine they were fitted on.

/// A CPU class: the vendor, family and model of the first CPU and the number of
/// logical CPUs, read from `/proc/cpuinfo` (Linux).
///
/// Unlike the kernel dispatch a machine takes, this is the key a provider's
/// offline-fitted rows are recorded under: two SKUs of one microarchitecture
/// with different core counts are different classes. A provider compares it
/// against the classes its rows were fitted on (see [`crate::ItemFit`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CpuClass {
    /// `vendor_id`, e.g. `GenuineIntel`.
    pub vendor: String,
    /// `cpu family`.
    pub family: u32,
    /// `model`.
    pub model: u32,
    /// Logical CPUs: the number of `processor` entries.
    pub cpus: usize,
}

impl CpuClass {
    /// This host's class, read once per process; `None` off Linux or when
    /// `/proc/cpuinfo` cannot be read or parsed.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_cost::host::CpuClass;
    ///
    /// // The same answer on every call.
    /// assert_eq!(CpuClass::detect(), CpuClass::detect());
    /// ```
    pub fn detect() -> Option<&'static CpuClass> {
        static CLASS: std::sync::OnceLock<Option<CpuClass>> = std::sync::OnceLock::new();
        CLASS.get_or_init(read_cpu_class).as_ref()
    }
}

#[cfg(target_os = "linux")]
fn read_cpu_class() -> Option<CpuClass> {
    parse_cpuinfo(&std::fs::read_to_string("/proc/cpuinfo").ok()?)
}

#[cfg(not(target_os = "linux"))]
fn read_cpu_class() -> Option<CpuClass> {
    None
}

/// The class from the text of `/proc/cpuinfo`: the first CPU's vendor, family
/// and model, and the number of `processor` entries.
#[cfg(any(target_os = "linux", test))]
fn parse_cpuinfo(text: &str) -> Option<CpuClass> {
    let field = |name: &str| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name).then(|| value.trim())
        })
    };
    let cpus = text
        .lines()
        .filter(|line| {
            line.split_once(':')
                .is_some_and(|(k, _)| k.trim() == "processor")
        })
        .count();
    Some(CpuClass {
        vendor: field("vendor_id")?.to_owned(),
        family: field("cpu family")?.parse().ok()?,
        model: field("model")?.parse().ok()?,
        cpus,
    })
}

#[cfg(test)]
mod tests;
