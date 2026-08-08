"""Bind the one-time evidence refresh to the exact first public dataset commit.

The initial publication completed all repository and smoke verification before a
later Git-only evidence update was ready. This immutable contract prevents the
authorized follow-up from adopting manual Hub mutations or rewriting historical
evidence: only the retrospective and derived paper PDF may differ.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

# This is the anonymously verified evidence revision from the successful publication.
PRE_REFRESH_EVIDENCE_REVISION: Final = "d6223aeac48c87faca586efec21cb48221f2640c"
# The retrospective is correctable public synthesis; the PDF is its derived paper view.
REFRESHABLE_EVIDENCE_PATHS: Final = frozenset(
    {
        "EXPERIMENTS.md",
        "output/pdf/teaching-one-synthetic-fact-qwen35.pdf",
    }
)
# Every other same-name byte must remain exactly as anonymously verified at the parent.
PRE_REFRESH_EVIDENCE_FILES: Final = MappingProxyType(
    {
        "EXPERIMENTS.md": "7a5ac961bdf4c56fc5caffa9e2335146d4b1fe1c1c888c82d1b89e2faabe714f",
        "LICENSE": "5b4a95d82199749043a8d826da776e909ffd03363579bf9d6c90931a88955f96",
        "README.md": "7ef21ddd7a9a13637d32350f010a9bf828f6e007e3bd62b84976c5158d50d1fe",
        "manifest.json": "28b4d5f50a39257d71b2b3e89e0468eff0bdb336bc16ebd9455cdbeec38cfe5f",
        "output/pdf/teaching-one-synthetic-fact-qwen35.pdf": "7fad5ad1509989ae71a9807490d2d88e054d0f5706a65fae9aa7344c10c7cf53",
        "paper/evidence/authoring-disclosure.json": "11ed06024e1968ec0c72e2276f15185e03f119831189d9989ea696fc18c38cc2",
        "publication_inventory.json": "090c0b2fe75f6d2a18990dd9f9dc974ff602c379daa162188e475e02abeabb99",
        "reports/evaluation-20260731T053727489078Z.json": "5b6c796b4e474f1ed9991e336908b6f417d290291bc6db0bfa1d746695a11299",
        "reports/evaluation-20260731T053727489078Z.md": "05fde5d40dd06495e84cbaafe43cb6f4b7351b1c40727fdb4c0879ff0135cb7a",
        "reports/evaluation-20260731T060709715986Z.json": "2ed534f6a890677132980ed96c8cee51fcf2c6cee9183049a264828333bc802c",
        "reports/evaluation-20260731T060709715986Z.md": "79bb1a0d8c39e69f64c47e14d44168c2427d16d9dfe8247f7e128758d1de788e",
        "reports/evaluation-20260731T075738153557Z.json": "21e9e1b05804da55be54acecc8d790760826e7531bc7bdc0162083e0d9607839",
        "reports/evaluation-20260731T075738153557Z.md": "07efc8ed7a42a2bb7e3ed8444daa633f2e110ca9e134aa50b7495810ae8c0c43",
        "reports/evaluation-20260731T205057425949Z.json": "b3eecffec00884c62c9b5557552327a19584c728eafb5195dfe2b57c65ac9ff1",
        "reports/evaluation-20260731T205057425949Z.md": "d63921095e36abbd2eb0fa5c8e7927a9e7c214957a7640e4e142867bdda8cc5a",
        "reports/evaluation-20260731T211115088822Z.json": "891af620a0e487d9dc5791860e6145b79fa32aaae0ea92a9efc04e827997eeed",
        "reports/evaluation-20260731T211115088822Z.md": "894d46a1c10e68fc75db4f7ec97d5a5d83753bcf238b607745a859625af14bc0",
        "reports/evaluation-20260731T222110336918Z.json": "36fabc4a7b8231e82d6fd38447c53f825cf428982f8cc56cc5b74191aa68fce8",
        "reports/evaluation-20260731T222110336918Z.md": "ef838ccdcd78e2b0cf20e8b309484dc52106b946653e95ef28528e28212040f1",
        "reports/evaluation-20260731T232459751161Z.json": "c4c45b992b31b26fd287f7e1ceac9dbd321e7f91d0371c6f759bb016d1f03518",
        "reports/evaluation-20260731T232459751161Z.md": "b581533abed7d6cbf25e53ef9a0833a4fe1092a7093e25ad54c4bb27ac1e5e9d",
        "reports/evaluation-20260801T002847084442Z.json": "e6ff6bc89173f3e4a495e44abdbe20f637d993819524f8dd2775a677f3912395",
        "reports/evaluation-20260801T002847084442Z.md": "4bfc5a76ddd8900c494dab044d1a951770c68ff2c7598abdee94a3b7f654d43c",
        "reports/experiments/README.md": "f4b907194cb52eb90ca6bd1e51bbe1b81d637e52a2441b38c22019994b247b5b",
        "reports/experiments/conservative.md": "9bb42bda507cae44bf638c5294cb7b51599f2618b1179adf687b28b7c710181b",
        "reports/experiments/expanded.md": "d45aba3c4ad117e29f155a92151667a124e51d87dc060f16b465a1ae8da6229b",
        "reports/experiments/minimal_pair_conservative.md": "14aef166ef0c36e14348a43d9b055212103d0c9cf2c0a157b3c569bcf96e8437",
        "reports/experiments/minimal_pair_expanded.md": "ea9a25aad82482c2450279d4e213b974b64bd878c526b88fd5d3d559e5d9d065",
        "reports/experiments/minimal_pair_primary.md": "939fef9c6b04843dd466873555e2ad97f6cd7892d2c8712f4e5fb060edd56561",
        "reports/experiments/paper_single_edit.md": "19c2a13c27f82817543870ce38ae3bda9e0416cf1ddabfb20e76c69203c92d05",
        "reports/experiments/primary.md": "15c6532e8e164ac76914b00674543842a1584859eea2c496aa55990e920b5fe5",
        "reports/experiments/semantic_specificity.md": "53cfaa7e9f91b03ac37b44d65a7e4f55f41bf2a46797af6b68ac30a72b6305e6",
        "reports/experiments/semantic_specificity_gentle.md": "81d79348176d58cc4fcf6264d75ed219cbc70683dfd8d0823f802eb771d6ca12",
        "reports/runs/README.md": "e9f9e077f46a196e37c43d780592507799ba57d60121f61402769325dab7360b",
        "reports/runs/conservative.md": "6656dd466b1e90d635187f2f645ccaecb30ee44e7c02d3d7a977b100428d68bd",
        "reports/runs/expanded.md": "dcf56f11e8b891c4c992e1740d5c5211d48803360327c789e85a41138a6465ce",
        "reports/runs/minimal_pair_conservative.md": "54dc52638e0651716eb35b1c6cbe5b5ed6d516c060946d796ded71389cd8de9b",
        "reports/runs/minimal_pair_expanded.md": "5b14eb715d64bb222fee3e0ca1c77fc37d83c9f4a8814e00360b734060884ed4",
        "reports/runs/minimal_pair_primary.md": "ed33d99b23900f3a820b54417794b817f1f23528a237ddf7864376ee035e1997",
        "reports/runs/paper_single_edit.md": "e0695138cd7dfd228f0a540080344025b34c70f87a40ce0a2a143d525b477edd",
        "reports/runs/primary.md": "418d9fa2c96a94c17dfccb267342e4b71dd6e5d735c0a22f84704c373ddc63eb",
        "reports/runs/semantic_specificity.md": "fc0789e768f4718fe854a507dbf8dd270f4384fd357ab16999a14441dc9c74f1",
        "reports/runs/semantic_specificity_gentle.md": "2406c651a029ad8323318f15b4b60cf831be4adf42ad61bc6d58fcf661c51a36",
    }
)
# This full target map binds even the two mutable paths to reviewed post-merge bytes.
FINAL_REFRESHED_EVIDENCE_FILES: Final = MappingProxyType(
    {
        **PRE_REFRESH_EVIDENCE_FILES,
        "EXPERIMENTS.md": "137de3ed7930a43b21b29ab66392309f1e587d1f6823d96ded7ef45b193b448d",
        "output/pdf/teaching-one-synthetic-fact-qwen35.pdf": "85fbff3a8bb5e82da28bcf7e9354779f9f389310161aeb16c040b5ba87d202a5",
    }
)

if len(PRE_REFRESH_EVIDENCE_FILES) != 43:
    raise RuntimeError("pre-refresh evidence contract must contain exactly 43 files")
if not REFRESHABLE_EVIDENCE_PATHS < set(PRE_REFRESH_EVIDENCE_FILES):
    raise RuntimeError("refreshable evidence paths must be a strict contract subset")
if set(FINAL_REFRESHED_EVIDENCE_FILES) != set(PRE_REFRESH_EVIDENCE_FILES):
    raise RuntimeError("final evidence contract must preserve the exact file set")
if {
    path
    for path in PRE_REFRESH_EVIDENCE_FILES
    if PRE_REFRESH_EVIDENCE_FILES[path] != FINAL_REFRESHED_EVIDENCE_FILES[path]
} != REFRESHABLE_EVIDENCE_PATHS:
    raise RuntimeError("only reviewed evidence paths may differ in the final contract")
