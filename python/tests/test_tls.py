from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from scylla.session_builder import SessionBuilder
from scylla.tls import TlsContextBuilder
from tests.helpers.ccm import (  # pyright: ignore[reportMissingTypeStubs]
    create_scylla_cluster,
    get_contact_points,
    start_cluster,
    stop_and_remove_cluster,
)


def _generate_private_key() -> RSAPrivateKey:
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def _certificate_name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                common_name,
            )
        ]
    )


def _generate_ca_certificate(
    private_key: RSAPrivateKey,
) -> x509.Certificate:
    now = datetime.now(timezone.utc)
    name = _certificate_name("Test CA")

    return (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(
                ca=True,
                path_length=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )


def _generate_leaf_certificate(
    *,
    common_name: str,
    private_key: RSAPrivateKey,
    ca_certificate: x509.Certificate,
    ca_private_key: RSAPrivateKey,
    usage: x509.ObjectIdentifier,
    subject_alternative_name: x509.GeneralName | None = None,
) -> x509.Certificate:
    now = datetime.now(timezone.utc)

    builder = (
        x509.CertificateBuilder()
        .subject_name(_certificate_name(common_name))
        .issuer_name(ca_certificate.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(
                ca=False,
                path_length=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([usage]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_private_key.public_key()),
            critical=False,
        )
    )

    if subject_alternative_name is not None:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([subject_alternative_name]),
            critical=False,
        )

    return builder.sign(
        ca_private_key,
        hashes.SHA256(),
    )


def _write_private_key(
    path: Path,
    private_key: RSAPrivateKey,
) -> None:
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_certificate(
    path: Path,
    certificate: x509.Certificate,
) -> None:
    path.write_bytes(
        certificate.public_bytes(
            serialization.Encoding.PEM,
        )
    )


@pytest.fixture(scope="module")
def generated_certs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """
    Generate an ephemeral CA, server identity, and client identity.

    All files are stored in pytest's temporary directory and are removed
    automatically after the test run.
    """
    certs_dir = tmp_path_factory.mktemp("tls-certs")

    ca_private_key = _generate_private_key()
    ca_certificate = _generate_ca_certificate(ca_private_key)

    server_private_key = _generate_private_key()
    server_certificate = _generate_leaf_certificate(
        common_name="127.0.0.1",
        private_key=server_private_key,
        ca_certificate=ca_certificate,
        ca_private_key=ca_private_key,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        subject_alternative_name=x509.IPAddress(ip_address("127.0.0.1")),
    )

    client_private_key = _generate_private_key()
    client_certificate = _generate_leaf_certificate(
        common_name="Python Driver",
        private_key=client_private_key,
        ca_certificate=ca_certificate,
        ca_private_key=ca_private_key,
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
    )

    _write_certificate(
        certs_dir / "ca.crt",
        ca_certificate,
    )
    _write_certificate(
        certs_dir / "server.crt",
        server_certificate,
    )
    _write_private_key(
        certs_dir / "server.key",
        server_private_key,
    )
    _write_certificate(
        certs_dir / "client.crt",
        client_certificate,
    )
    _write_private_key(
        certs_dir / "client.key",
        client_private_key,
    )

    return certs_dir


def test_tls_context_from_pem(generated_certs: Path):
    """Test that we can parse valid PEM bytes."""
    ca_bytes = (generated_certs / "ca.crt").read_bytes()

    context = TlsContextBuilder().set_verify_peer(False).load_verify_locations(cadata=ca_bytes).build()

    assert context is not None


def test_tls_context_from_files(generated_certs: Path):
    """Test the file loading convenience helper."""
    ca_path = generated_certs / "ca.crt"

    context = TlsContextBuilder().set_verify_peer(True).load_verify_locations(cafile=ca_path).build()

    assert context is not None


def test_tls_context_builder_methods_are_chainable():
    """Test that public TLS builder methods return the builder."""
    builder = TlsContextBuilder()

    assert builder.set_verify_peer(False) is builder


def test_tls_context_missing_key_for_mtls_raises(generated_certs: Path):
    """Test that providing a cert without a key raises our custom TlsError."""
    cert_bytes = (generated_certs / "client.crt").read_bytes()

    with pytest.raises(
        ValueError,
        match="Either keyfile or keydata must be provided",
    ):
        TlsContextBuilder().load_cert_chain(
            certdata=cert_bytes,
        )


def test_session_builder_tls_chaining(generated_certs: Path):
    """Test that the builder methods return self and store the config."""
    builder = SessionBuilder()

    context = TlsContextBuilder().load_verify_locations(cafile=generated_certs / "ca.crt").build()

    # Test attaching
    assert builder.tls_context(context) is builder

    # Test detaching
    assert builder.tls_context(None) is builder


# --- Integration Tests ---


@pytest.fixture(scope="module")
def tls_ccm_cluster(
    generated_certs: Path,
) -> Generator[list[tuple[str, int]], None, None]:

    tls_config = {
        "client_encryption_options": {
            "enabled": True,
            "require_client_auth": True,
            "certificate": str(generated_certs / "server.crt"),
            "keyfile": str(generated_certs / "server.key"),
            "truststore": str(generated_certs / "ca.crt"),
        }
    }

    cluster = create_scylla_cluster(
        name="tls_cluster",
        scylla_version="release:6.2.2",
        nodes=1,
        config=tls_config,
    )

    start_cluster(cluster)

    try:
        yield get_contact_points(cluster)
    finally:
        stop_and_remove_cluster(cluster)


@pytest.mark.asyncio
@pytest.mark.requires_ccm
@pytest.mark.parametrize("verify_peer", [False, True])
async def test_tls_connection_success(
    tls_ccm_cluster: list[tuple[str, int]],
    generated_certs: Path,
    verify_peer: bool,
):
    """Tests that the driver can securely connect to a TLS-enabled cluster."""
    ca_path = generated_certs / "ca.crt"
    client_cert_path = generated_certs / "client.crt"
    client_key_path = generated_certs / "client.key"

    tls_config = (
        TlsContextBuilder()
        .set_verify_peer(verify_peer)
        .load_verify_locations(cafile=ca_path)
        .load_cert_chain(certfile=client_cert_path, keyfile=client_key_path)
        .build()
    )

    builder = SessionBuilder().contact_points(tls_ccm_cluster).tls_context(tls_config)

    session = await builder.connect()

    # Prove we can talk to the database
    result = await session.execute("SELECT cluster_name FROM system.local")
    row = await result.first_row()

    assert row is not None
