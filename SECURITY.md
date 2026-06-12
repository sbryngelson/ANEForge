# Security Policy

ANEForge runs in an ordinary user process. It needs no entitlement and makes no
changes to System Integrity Protection. It builds a native dispatch shim from
source (`aneforge/_lib/`) and calls private, undocumented Apple framework symbols
to reach the Neural Engine.

## Reporting

Report security issues privately, not in a public issue:

- GitHub private vulnerability reporting, under this repository's **Security** tab.
- Or email **sbryngelson@gmail.com**, subject "ANEForge security".

Include the chip, the macOS version, and the smallest reproduction you have. This
is a single-maintainer project; responses are best effort.

## Scope

In scope: the `aneforge` package and the dispatch shim source in this repository.
For example, memory safety in the native shim, a graph input that drives unsafe
behavior, or a packaging concern.

Out of scope here, and handled through coordinated disclosure with the vendor
instead: issues in macOS or Apple's system services rather than in this code.

## Coordinated disclosure with Apple

ANEForge calls Apple's private frameworks and dispatches work to a system service.
An issue that affects macOS or an Apple system component, rather than this
project's own code, should also go to Apple Product Security
(https://security.apple.com, product-security@apple.com). This project follows
coordinated disclosure and will not publish technical details of an OS-level or
system-service issue before the vendor has addressed it.

## Supported versions

Pre-1.0 research software. Fixes land on the latest release and main; no backports.

| Version        | Supported |
| -------------- | :-------: |
| latest release | yes       |
| older          | no        |

## Private symbols

ANEForge depends on undocumented symbols that Apple may change or remove without
notice, and behavior varies across chips and macOS versions. Verify it on your own
OS version before relying on it, and keep the package's default safety behavior
enabled.
