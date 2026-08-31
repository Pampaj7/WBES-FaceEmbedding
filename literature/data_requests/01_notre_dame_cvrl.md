# Notre Dame CVRL — richiesta d'accesso

**A:** `cvrl@nd.edu`
**Oggetto:** `Dataset access request — FRGC v2.0, ND-2006, 3D-TEC, Collection D`
**Da:** indirizzo istituzionale DTU
**Allegato:** accordo di licenza CVRL firmato da CHI ASSUME IMPEGNI LEGALI PER DTU

> Nota procedurale: CVRL richiede che l'accordo sia firmato da chi è autorizzato ad assumere
> impegni legali per l'ente, **non** dal supervisore in quanto tale. Consegna via Globus.

---

Dear CVRL Data Distribution Team,

I am writing to request access to the FRGC v2.0, ND-2006, 3D-TEC and ND Collection D datasets
for academic research at the Technical University of Denmark (DTU), Department of DIPARTIMENTO.

**Research context.** Our group studies how 3D face reconstruction methods are *evaluated*.
Current benchmarks score a reconstruction by its per-pair surface error after registration to a
ground-truth scan. We are investigating a complementary question: whether a distance measure
preserves the correct *ordering of identities* — that is, whether two scans of the same person
are placed closer together than two scans of different people, when the meshes involved differ
in vertex count, connectivity and surface support.

**Why these datasets specifically.** Our work so far has necessarily used synthetic meshes
generated from a morphable model, where every "topology" is one we produced ourselves. This is
a real limitation, and it is why we are writing. In FRGC v2.0 and ND-2006, each scan is an
independent acquisition, so mesh resolution, connectivity and valid-region masks genuinely vary
between captures of the same subject — variation that exists in the world rather than variation
we manufactured. Equally important, the subject identifier is an annotation independent of the
geometry, which lets us evaluate identity ordering against a label we did not derive ourselves.

3D-TEC is of particular interest. Identical twins have nearly identical facial geometry and
distinct identities, which makes them a decisive test of whether a distance measure tracks
identity or merely shape. We are not aware of another public resource that permits this test.

**Intended use.** Non-commercial academic research only: evaluating and comparing distance
measures for 3D face data, and reporting aggregate results in peer-reviewed publications. We
will not attempt to identify subjects beyond the identifiers you provide, will not redistribute
the data or any derivative from which the original scans could be reconstructed, and will
restrict access to named researchers covered by the signed agreement. Any published artifacts
will contain aggregate statistics and model parameters only. We are glad to follow any specific
attribution or embargo requirements you set.

**Personnel.** The signed agreement covers NOME SUPERVISORE (principal investigator) and
Leonardo Pampaloni (PhD researcher, who will handle the data day to day). Storage will be on
DTU-managed infrastructure with access limited to those named.

The countersigned license agreement is attached. Please let me know if you need anything
further, or if the intended use above should be described differently to fit your terms.

With thanks for maintaining these resources,

NOME
Department of DIPARTIMENTO, Technical University of Denmark
INDIRIZZO EMAIL ISTITUZIONALE
