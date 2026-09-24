"""Internal helpers for host/URL extraction and matching (egress guards).

Shared by ``tool_host_within`` and ``no_exfil_links``. Not part of the public
API — import paths here may change without notice.
"""

from __future__ import annotations

import re
import socket


def _as_ip(host: str):
    import ipaddress

    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _extract_host(value: str):
    """Pull the host out of a URL or bare host[:port][/path]; lowercased, no brackets."""
    from urllib.parse import urlsplit

    v = value.strip()
    if not v:
        return None
    try:
        netloc_form = v if ("://" in v or v.startswith("//")) else "//" + v
        host = urlsplit(netloc_form).hostname
    except ValueError:
        return None
    return host.lower() if host else None


def _url_like(value: str) -> bool:
    """Heuristic: is this string worth treating as a URL/host for egress checks?"""
    v = value.strip()
    if not v or any(ch.isspace() for ch in v):
        return False
    if "://" in v:
        return True
    head = v.split("/")[0]
    hostpart = head.rsplit(":", 1)[0].strip("[]")
    if hostpart == "localhost":
        return True
    if _as_ip(hostpart) is not None:
        return True
    return ("." in hostpart) and all(ch.isalnum() or ch in ".-" for ch in hostpart)


def _host_matches(host: str, patterns: list[str]) -> bool:
    """True if host matches any glob host pattern or IP/CIDR in ``patterns``."""
    import ipaddress
    from fnmatch import fnmatch

    host_ip = _as_ip(host)
    for p in patterns:
        pl = p.lower()
        if host_ip is not None:
            try:
                net = ipaddress.ip_network(p, strict=False)
            except ValueError:
                net = None
            if net is not None and host_ip.version == net.version and host_ip in net:
                return True
        if fnmatch(host, pl):
            return True
    return False


def _canonical_hosts(host: str) -> list[str]:
    """Every address a client might connect to for one literal host.

    - A trailing root dot (``localhost.``) resolves identically, so strip it.
    - Legacy numeric IPv4 forms - decimal ``2130706433``, hex ``0x7f000001``,
      octal ``0177.0.0.1``, short ``127.1`` - are rejected by :mod:`ipaddress`
      but accepted by libc's ``inet_aton``, which many HTTP clients use. A guard
      that only understands dotted-quad treats them as hostnames and lets
      ``http://2130706433/`` (loopback) straight through. Include the address
      ``inet_aton`` yields alongside the literal, so both get checked.
    """
    h = host.strip().lower().rstrip(".")
    if not h:
        return []
    out = [h]
    if _as_ip(h) is None and re.fullmatch(r"[0-9a-fx.]+", h):
        try:
            out.append(socket.inet_ntoa(socket.inet_aton(h)))
        except OSError:
            pass
    return out


def _candidate_hosts(value: str) -> list[str]:
    """All hosts a real client might connect to for a URL or bare host string.

    Parsers disagree, and every recent egress-guard CVE of this class came from
    the validator reading a URL differently from the client that fetched it.
    RFC 3986 treats ``\\`` as an ordinary character, so
    ``http://evil.com\\@good.com/`` is host ``good.com``; WHATWG (browsers,
    Node, many agent runtimes) treats it as ``/``, giving host ``evil.com``.
    Rather than bet on one parser, return every interpretation - callers must
    reject the value if ANY of them is disallowed.
    """
    v = value.strip()
    if not v:
        return []
    views = [v]
    if "\\" in v:
        views.append(v.replace("\\", "/"))
    hosts: list[str] = []
    for view in views:
        literal = _extract_host(view)
        if not literal:
            continue
        for candidate in _canonical_hosts(literal):
            if candidate not in hosts:
                hosts.append(candidate)
    return hosts


def _is_private_host(host: str) -> bool:
    """True for localhost or a private/loopback/link-local/reserved IP literal."""
    host = host.lower().rstrip(".")
    # RFC 6761: ``localhost`` and every ``*.localhost`` name are loopback.
    if host == "localhost" or host.endswith(".localhost"):
        return True
    ip = _as_ip(host)
    if ip is None:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
