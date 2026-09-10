# Release integrity and serving verification

This is a publication contract, not a claim that an existing image satisfies it.
A release must identify its immutable container digest and provide the evidence
below before it can be considered verified. A tag alone is not an identity.

## Required release evidence

For each published image index, record:

- The complete source commit SHA, public source location, and whether the source
  tree was clean. Uncommitted patches require a separately checksummed source
  artifact; they must not be described as the unmodified commit.
- Every build-stage base image by digest, including builder and runtime bases.
- Target platforms, build targets, compiler/toolchain versions, backend choices,
  and effective build flags. Record platform-specific differences explicitly.
- The index digest and each platform manifest digest, with the corresponding
  SBOM and provenance subjects. Evidence for one architecture is not evidence
  for another.
- An SBOM covering the shipped runtime filesystem and provenance identifying
  source, builder, inputs, and invocation. Bind these records to the actual
  published digest, not only a mutable tag or a nearby download filename.
- An authenticated publisher statement covering these subjects, plus the public
  verification policy: trusted key or exact identity and issuer, allowed source
  repository/ref, builder, and required platforms. Credentials and private
  infrastructure details must never appear in build arguments or attestations.

BuildKit can attach SBOM and provenance metadata using `--sbom=true` and
`--provenance=mode=max`. These attachments alone do not authenticate a publisher.
See [Docker build attestations](https://docs.docker.com/build/metadata/attestations/).
A successful build or presence of metadata does not satisfy the verification
policy by itself. Missing, mismatched, or unauthenticated evidence blocks release.

## Verification by an independent consumer

Obtain the expected digest and trust policy through an authenticated release
channel. Do not adopt a key merely because the image itself supplied it.
Use the digest reference for inspection, verification, and execution:

```bash
IMAGE='registry.example.org/project/server@sha256:<expected-index-digest>'
docker buildx imagetools inspect "$IMAGE"
docker buildx imagetools inspect "$IMAGE" --raw
docker buildx imagetools inspect "$IMAGE" --format '{{json .Provenance}}'
docker buildx imagetools inspect "$IMAGE" --format '{{json .SBOM}}'
```

These are inspection commands, documented by
[Docker imagetools inspect](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/).
Check all required platforms, follow their manifest and attestation references,
and compare each statement's subject with the corresponding digest. Compare
source SHA, base inputs, flags, and builder against the release policy. Save the
raw evidence and verification output with the release record.

Verify the authenticated statement using the publisher's declared signature
format and an independently trusted key or identity. For Cosign attestations,
use `cosign verify-attestation` with that trust policy and the appropriate
predicate type; validate the decoded subject and predicate as well as the
signature. Native BuildKit attachments are not automatically Cosign signed
attestations. See [Cosign attestation verification](https://docs.sigstore.dev/cosign/verifying/attestation/).
If no supported authenticated statement exists, report verification incomplete.

## Model identity is separate

Container identity covers software and its runtime filesystem. Model identity
must have a separate record: public origin and revision, exact filenames,
quantization/format, byte lengths, and checksums for every loaded file, including
shards and adapters. Record tokenizer and chat-template choices where external.
A model alias returned by an API is not checksum evidence. A healthy image does
not prove which model is mounted; unchanged model bytes do not prove image
integrity. Link both records in a serving result without substituting either.

## Static executable check

Run from the checkout; Python 3 is the only runtime dependency:

```bash
ci/check-image-entrypoints.sh --target server docker/*.Containerfile
ci/check-image-entrypoints.sh
```

The first checks a named stage and its ancestors; the second checks every stage
of every Containerfile under `docker/`. Entrypoint executables and copied binary
artifacts must match in-tree CMake executable declarations. Referenced shell or
Python scripts must exist in the tree. External binaries without build targets
fail. Enumerate executable copies explicitly: unresolved variables, globs, and
bulk binary-directory copies fail closed. Exec-form entrypoints are required.

The checker resolves literal declarations and local `set()` variables; it does
not implement all CMake or Docker syntax. A declared target may be disabled by
configuration. The JSON explicitly reports static declaration scope: this does
not prove that the selected build produced, copied, or can execute an artifact.
Build and run the final image on its intended platform to establish those facts.

## Real serving smoke check

Start the exact digest-selected image with separately verified model artifacts
on a suitable host, then run:

```bash
SMOKE_MODEL='<served-model-id>' ci/smoke-serve.sh \
  http://127.0.0.1:8080 --min-tps 1
```

Use only an endpoint reserved for this test; do not probe a shared single-slot
service. Compilation belongs on a build host. Stop an owned test server
normally after collecting evidence.

The URL may include `/v1`. `SMOKE_MODEL` selects a model; if omitted, discovery
requires exactly one model. `SMOKE_API_KEY` supplies an optional bearer token;
`SMOKE_TIMEOUT` sets the positive socket timeout in seconds (default 60).
Redirects are rejected. URLs containing credentials, queries, or fragments are
rejected. Output omits endpoint, key, model name, and completion text.

The script sends one non-streaming chat completion with a 32-token limit. Success
requires nonempty generated text and a positive integer `usage.completion_tokens`.
It emits one JSON result with UTC start/end timestamps, request duration, token
count, text length, and throughput. Exit zero means those checks and the optional
minimum passed; malformed responses, HTTP errors, timeouts, missing usage, and
below-threshold results exit nonzero. Consumers must also check `ok`.

Throughput is completion tokens divided by total chat-request time, including
prefill and network overhead; it is not decode-only speed. Discovery contributes
to total elapsed time but not request throughput. A tiny fixture proves request
handling only. Record model checksums, image digest, platform/backend, command,
script revision, and output beside the measurement. A completion from a standalone
server does not establish that a published container works. Health endpoints,
open ports, and declared capabilities are never completion evidence.
