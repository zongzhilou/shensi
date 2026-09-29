# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.
from dataclasses import dataclass

from megatron_ext.core.models.shensi.shensi_layer_specs import (
    get_shensi_decoder_block_spec,
    get_shensi_mtp_layer_spec,
)
from megatron_ext.core.models.shensi.shensi_model import ShensiModel
from megatron_ext.core.transformer.shensi.transformer_config import ShensiTransformerConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class ShensiModelProvider(ShensiTransformerConfig, GPTModelProvider):
    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> ShensiModel:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )
        from megatron.core.utils import init_method_normal, scaled_init_method_normal
        from megatron.training.vocab_utils import calculate_padded_vocab_size

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)
        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(self.init_method_std, self.num_layers)
        if self.embedding_init_method is None:
            self.embedding_init_method = init_method_normal(self.embedding_init_method_std or self.init_method_std)
        transformer_layer_spec = get_shensi_decoder_block_spec(
            config=self, use_transformer_engine=True, vp_stage=vp_stage
        )
        mtp_block_spec = None
        if self.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                self,
                get_shensi_mtp_layer_spec(self),
                use_transformer_engine=True,
                vp_stage=vp_stage,
            )
        vp_size = self.virtual_pipeline_model_parallel_size
        pg_collection = self._pg_collection
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
        self._vp_stage = vp_stage
        padded_vocab_size = self.vocab_size
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        return ShensiModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rope_scaling=self.rope_scaling,
            rope_scaling_factor=self.rope_scaling_factor,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            mtp_block_spec=mtp_block_spec,
        )
