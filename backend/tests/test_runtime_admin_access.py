from gateway.admin_access import enrich_runtime_admin_role, is_runtime_admin_email


def test_runtime_admin_allowlist_email_is_authoritative():
    assert is_runtime_admin_email("oomkaragarkhed0710@gmail.com")
    assert is_runtime_admin_email("OOMKARAGARKHED0710@GMAIL.COM")
    assert not is_runtime_admin_email("someone@example.com")


def test_runtime_admin_role_enrichment_is_email_based():
    user = {
        "user_id": "u1",
        "email": "oomkaragarkhed0710@gmail.com",
        "role": "authenticated",
    }
    enriched = enrich_runtime_admin_role(user)
    assert enriched["role"] == "admin"
    assert enriched["runtime_admin"] is True

    regular = enrich_runtime_admin_role({"user_id": "u2", "email": "regular@example.com", "role": "admin"})
    assert regular["runtime_admin"] is False
