#!/bin/bash

DEPFILE="${DEPFILE:=./dependencies.txt}"

set -e

main()
{
    local proxy="${HTTP_PROXY:-$http_proxy}"

    if [ -n "$proxy" ]
    then
        printf "Proxy %s was found, creating proxy.conf...\\n" "$proxy"
        printf 'Acquire::http::Proxy "%s";\n' "$proxy" >> /etc/apt/apt.conf.d/proxy.conf
    fi

    printf "Update apt sources...\\n"

    local err

    if ! err="$(apt-get --assume-yes update 2>&1)"
    then
        printf "Could not update because: '%s'\\n" "$err"
        exit 1
    fi

    mapfile -t < "$DEPFILE"

    local packages=""

    for line in "${MAPFILE[@]}"
    do
        [[ "$line" =~ ^[a-zA-Z0-9_] ]] && packages="$packages$line "
    done

    printf "Install packages %s...\\n" "$packages"

    #shellcheck disable=SC2086
    if ! err="$(apt-get --assume-yes install $packages 2>&1)"
    then
        printf "Could not install packages because: '%s'\\n" "$err"
        exit 1
    fi

    rm --recursive --force /var/lib/apt/lists/*

    exit 0
}

main
