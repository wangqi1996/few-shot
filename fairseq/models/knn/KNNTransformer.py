from typing import Optional, Dict, Any, Tuple, List

import torch
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model_architecture, register_model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.knn.KNNDatastore import KNN_Dstore
from fairseq.models.transformer import TransformerModel, TransformerDecoder, base_architecture, transformer_wmt19_de_en


@register_model("knn_transformer")
class KNNTransformer(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return KNNTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

class KNNTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(KNNTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.knn_datastore = KNN_Dstore(args, len(dictionary))
        self.lambda_value = self.knn_datastore.lambda_value

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        knn_prob = self.knn_datastore.forward(x)
        if not features_only:
            x = self.output_layer(x)

        extra['knn_prob'] = knn_prob
        return x, extra

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = net_output[0]
        nmt_porb = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        knn_prob = net_output[1]['knn_prob']

        probs = self.lambda_value * knn_prob + (1 - self.lambda_value) * nmt_porb
        if log_probs:
            return torch.log(probs)
        else:
            return probs

@register_model_architecture("knn_transformer", "knn_transformer")
def knn_base_architecture(args):
    base_architecture(args)

@register_model_architecture("knn_transformer", "knn_transformer_wmt19_de_en")
def knn_transformer_wmt19_de_en(args):
    transformer_wmt19_de_en(args)