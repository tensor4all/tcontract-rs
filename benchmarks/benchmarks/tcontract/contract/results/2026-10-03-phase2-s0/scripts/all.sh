#!/usr/bin/env bash
d=$(cd "$(dirname "$0")" && pwd)
"$d/session.sh" s1; "$d/session.sh" s2
