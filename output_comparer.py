import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from resprop_og import resprofify_bert  # your custom patch that injects ManualLinearFunction
import copy

def cosine_similarity(t1, t2):
    return F.cosine_similarity(t1.view(-1), t2.view(-1), dim=0).item()


def compare_parameters(param1, param2, name=""):
    cos_sim = F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
    max_diff = (param1 - param2).abs().max().item()
    print(f"{name}:")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  Max absolute difference: {max_diff:.6e}")

def test_linear_equivalence():
    # Setup
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"

    # Load base and custom models
    base_model = BertForSequenceClassification.from_pretrained(model_name, attn_implementation="eager", num_labels=2)
    custom_model = copy.deepcopy(base_model)
    # custom_model = resprofify_bert(custom_model, reuse_percentage=0)
    base_model.to(device).eval()
    custom_model.to(device).eval()

    # Choose a linear layer to compare — e.g., output projection of layer 0
    for i, layer in enumerate(base_model.bert.encoder.layer):
        base_att = base_model.bert.encoder.layer[i].attention.self
        custom_att = custom_model.bert.encoder.layer[i].attention.self
        base_linear = base_att.key
        custom_linear = custom_att.key

        compare_parameters(base_linear.weight.data, custom_linear.weight.data, name="Layer 0 - output.dense.weight")

        # Compare bias
        if base_linear.bias is not None:
            compare_parameters(base_linear.bias.data, custom_linear.bias.data, name="Layer 0 - output.dense.bias")
                # Ensure weights are identical

        # Create dummy input
        batch_size = 128
        seq_len = 128
        hidden_dim = base_linear.in_features
        dummy_input = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

        # Run through both layers
        with torch.no_grad():
            base_out = base_linear(dummy_input)
            custom_out = custom_linear(dummy_input)

        # Compare outputs
        cos_sim = cosine_similarity(base_out, custom_out)
        max_diff = (base_out - custom_out).abs().max().item()
        # print('layer', i)
        # print(f"Cosine similarity: {cos_sim:.8f}")
        # print(f"Max absolute difference: {max_diff:.6e}")

if __name__ == "__main__":
    test_linear_equivalence()