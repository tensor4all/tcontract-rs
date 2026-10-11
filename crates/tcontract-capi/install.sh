#!/usr/bin/env bash
#
# Install the built TAPP shared library, its headers, a pkg-config file and the
# licences into a prefix.
#
#   crates/tcontract-capi/install.sh --prefix=/opt/tapp
#
# Why this exists rather than a line of `cp` in each place that needs one: the
# same layout has to come out of a manual site install, of CI, and of the
# BinaryBuilder recipe in Yggdrasil, and three definitions of a
# layout are three definitions that drift. `examples/c-consumer` has a
# `TAPP_PREFIX` mode that consumes exactly what this produces, so the layout is
# tested rather than asserted.
#
# It does not build anything and it does not rewrite binaries. The SONAME and the
# Mach-O install name are set at *link* time -- see the Yggdrasil recipe and the
# "Installing" section of `examples/c-consumer/README.md` -- because rustc emits
# neither for a `cdylib`, and rewriting an ELF afterwards needs `patchelf
# --page-size 65536` on every 64 KiB-page architecture. This script checks for
# them and warns; it will not silently paper over a build that lacks them.

set -euo pipefail

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${crate_dir}/../.." && pwd)"

prefix=""
libdir=""
includedir=""
artifacts="${repo_root}/target/release"
version=""
licenses=1

usage() {
    cat >&2 <<EOF
usage: install.sh --prefix=DIR [options]

  --prefix=DIR      installation prefix (required)
  --libdir=DIR      library directory (default: PREFIX/lib, or PREFIX/bin for a DLL)
  --includedir=DIR  header directory (default: PREFIX/include)
  --artifacts=DIR   where cargo left the library (default: target/release)
  --version=X.Y.Z   version for the pkg-config file (default: read from Cargo.toml)
  --no-licenses     skip the licences, for a caller that installs them itself
                    under a different name -- BinaryBuilder's \`install_license\`
                    uses the package name (tcontract) where this
                    script uses the crate name (tcontract-capi), and its
                    auditor looks for the former
EOF
    exit 2
}

for arg in "$@"; do
    case "${arg}" in
        --prefix=*)     prefix="${arg#*=}" ;;
        --libdir=*)     libdir="${arg#*=}" ;;
        --includedir=*) includedir="${arg#*=}" ;;
        --artifacts=*)  artifacts="${arg#*=}" ;;
        --version=*)    version="${arg#*=}" ;;
        --no-licenses)  licenses=0 ;;
        -h|--help)      usage ;;
        *)              echo "install.sh: unknown argument '${arg}'" >&2; usage ;;
    esac
done

[[ -n "${prefix}" ]] || usage
[[ -d "${artifacts}" ]] || {
    echo "install.sh: no such artifact directory: ${artifacts}" >&2
    echo "            run 'cargo build --release -p tcontract-capi' first" >&2
    exit 1
}

# The version comes from this crate's manifest, which is the single source of
# truth the header macros and `TAPP_implementation_version()` are both checked
# against. Parsed rather than passed so a caller cannot supply a different one.
if [[ -z "${version}" ]]; then
    version="$(sed -n 's/^version = "\(.*\)"$/\1/p' "${crate_dir}/Cargo.toml" | head -n1)"
    [[ -n "${version}" ]] || { echo "install.sh: cannot read version from Cargo.toml" >&2; exit 1; }
fi

# Find the library by its filename rather than by guessing the host: this script
# runs under cross-compilation, where the host says nothing about the artifact.
# A `cdylib` is `libNAME.so` / `libNAME.dylib` / `NAME.dll` -- rustc's
# windows-gnu target spec sets an empty `dll_prefix`, so the DLL has no `lib`.
lib=""
for candidate in \
    "${artifacts}/libtcontract_capi.so" \
    "${artifacts}/libtcontract_capi.dylib" \
    "${artifacts}/tcontract_capi.dll"
do
    if [[ -f "${candidate}" ]]; then lib="${candidate}"; break; fi
done
[[ -n "${lib}" ]] || {
    echo "install.sh: no shared library in ${artifacts}" >&2
    echo "            expected libtcontract_capi.{so,dylib} or tcontract_capi.dll" >&2
    echo "            on musl this means the cdylib was dropped: build with" >&2
    echo "            RUSTFLAGS='-C target-feature=-crt-static'" >&2
    exit 1
}
libname="$(basename "${lib}")"

# A DLL is a runtime component and belongs next to the executables that load it,
# which is also where BinaryBuilder puts one.
if [[ -z "${libdir}" ]]; then
    case "${libname}" in
        *.dll) libdir="${prefix}/bin" ;;
        *)     libdir="${prefix}/lib" ;;
    esac
fi
[[ -n "${includedir}" ]] || includedir="${prefix}/include"

install -d "${libdir}" "${includedir}" "${prefix}/lib/pkgconfig"
[[ ${licenses} -eq 1 ]] && install -d "${prefix}/share/licenses/tcontract"

install -m 0755 "${lib}" "${libdir}/${libname}"
echo "installed ${libdir}/${libname}"

# The import library, which is how anything links against a Windows DLL.
if [[ "${libname}" == *.dll && -f "${artifacts}/libtcontract_capi.dll.a" ]]; then
    install -m 0644 "${artifacts}/libtcontract_capi.dll.a" \
                    "${prefix}/lib/libtcontract_capi.dll.a"
    echo "installed ${prefix}/lib/libtcontract_capi.dll.a"
fi

# The whole include tree: the pinned upstream TAPP headers (`tapp.h`, `tapp/`)
# and tcontract' own (`tcontract/`). Copied with their relative layout, which the
# headers' `#include` lines depend on.
(cd "${crate_dir}/include" && find . -type f \( -name '*.h' -o -name '*.md' \)) | while read -r header; do
    install -D -m 0644 "${crate_dir}/include/${header}" "${includedir}/${header#./}"
done
echo "installed headers under ${includedir}"

sed -e "s|@PREFIX@|${prefix}|g" \
    -e "s|@LIBDIR@|${libdir}|g" \
    -e "s|@INCLUDEDIR@|${includedir}|g" \
    -e "s|@VERSION@|${version}|g" \
    "${crate_dir}/tcontract_capi.pc.in" \
    > "${prefix}/lib/pkgconfig/tcontract_capi.pc"
echo "installed ${prefix}/lib/pkgconfig/tcontract_capi.pc"

if [[ ${licenses} -eq 1 ]]; then
    for licence in LICENSE-MIT LICENSE-APACHE; do
        install -m 0644 "${repo_root}/${licence}" \
                        "${prefix}/share/licenses/tcontract/${licence}"
    done
    echo "installed ${prefix}/share/licenses/tcontract/LICENSE-{MIT,APACHE}"
fi

# Report the shared-library identity rather than assume it. An ELF with no
# DT_SONAME makes every consumer record a bare filename; a Mach-O whose
# LC_ID_DYLIB is the build-tree path is not relocatable at all, and that is
# rustc's default for a cdylib, not an unusual accident.
case "${libname}" in
    *.so)
        if command -v readelf > /dev/null; then
            soname="$(readelf -d "${libdir}/${libname}" \
                      | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
            if [[ -z "${soname}" ]]; then
                echo "install.sh: warning: no DT_SONAME." >&2
                echo "            rustc does not emit one for a cdylib; rebuild with" >&2
                echo "            RUSTFLAGS='-C link-arg=-Wl,-soname,${libname}'" >&2
            else
                echo "soname: ${soname}"
            fi
        fi
        ;;
    *.dylib)
        if command -v otool > /dev/null; then
            id="$(otool -D "${libdir}/${libname}" | tail -n1 | tr -d '[:space:]')"
            case "${id}" in
                @rpath/*|"${libname}") echo "install name: ${id}" ;;
                *) echo "install.sh: warning: install name is '${id}', not relocatable." >&2
                   echo "            rebuild with RUSTFLAGS='-C link-arg=-Wl,-install_name,@rpath/${libname}'" >&2 ;;
            esac
        fi
        ;;
esac
