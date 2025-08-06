from cliport.agents.clipfit_conceptfusion import CLIPFitAgent
from cliport.agents.transporter import OriginalTransporterAgent
from cliport.agents.transporter import ClipUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipWithoutSkipsTransporterAgent
from cliport.agents.transporter import TwoStreamRN50BertUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipUNetTransporterAgent

from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamUntrainedRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import OriginalTransporterLangFusionAgent
from cliport.agents.transporter_lang_goal import ClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import ClipLingUNetTransporterAgentlesslayers
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetLatTransporterAgent

from cliport.agents.transporter_image_goal import ImageGoalTransporterAgent

from cliport.agents.transporter import TwoStreamClipUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgentlesslayers
from cliport.agents.transporter_lang_goal import TwoStreamClipFilmLingUNetLatTransporterAgent

from cliport.agents.transporter_cocoop import TwoStreamClipLingUNetLatTransporterAgentCoCoOp
from cliport.agents.transporter_cocoop import TwoStreamClipLingUNetLatTransporterAgentCoOp
from cliport.agents.transporter_cocoop import TwoStreamClipLingUNetLatTransporterAgentCoOpRN50
from cliport.agents.transporter_cocoop import TwoStreamClipLingUNetLatTransporterAgentCoCoOpRN50

from cliport.agents.clip_conceptfusion import ConceptFusionSam2
from cliport.agents.clip_conceptfusion import ConceptFusionAgent
from cliport.agents.clip_conceptfusion import ConceptFusionLarge
from cliport.agents.clip_conceptfusion import Sam2Clip
from cliport.agents.clip_conceptfusion import Sam2Clip_wo

from cliport.agents.clipfit_conceptfusion import CLIPFitAgent, CLIPFitAgent1, CLIPFitAllSimAgent, PretrainAgent, CLIPFitRealAgent

from cliport.agents.transporter_conceptfusion import TransFusionSam2

from cliport.agents.transporter_unetr import UnetrAgent


# from cliport.agents.chat import answer_question


names = {
         ################################
         ### CLIPort ###
         'cliport': TwoStreamClipLingUNetLatTransporterAgent,
         'two_stream_clip_lingunet_lat_transporter': TwoStreamClipLingUNetLatTransporterAgent,
         
         ################################
         ### CLIPort with less layers ###
         'cliport_less_layers': TwoStreamClipLingUNetLatTransporterAgentlesslayers,

         ################################
         ### Two-Stream Architectures ###
         # CLIPort without language
         'two_stream_clip_unet_lat_transporter': TwoStreamClipUNetLatTransporterAgent,

         # CLIPort without lateral connections
         'two_stream_clip_lingunet_transporter': TwoStreamClipLingUNetTransporterAgent,

         # CLIPort without language and lateral connections
         'two_stream_clip_unet_transporter': TwoStreamClipUNetTransporterAgent,

         # CLIPort without language, lateral, or skip connections
         'two_stream_clip_woskip_transporter': TwoStreamClipWithoutSkipsTransporterAgent,

         # RN50-BERT
         'two_stream_full_rn50_bert_lingunet_lat_transporter': TwoStreamRN50BertLingUNetLatTransporterAgent,

         # RN50-BERT without language
         'two_stream_full_rn50_bert_unet_transporter': TwoStreamRN50BertUNetTransporterAgent,

         # RN50-BERT without lateral connections
         'two_stream_full_rn50_bert_lingunet_transporter': TwoStreamRN50BertLingUNetTransporterAgent,

         # Untrained RN50-BERT (similar to untrained CLIP)
         'two_stream_full_untrained_rn50_bert_lingunet_transporter': TwoStreamUntrainedRN50BertLingUNetTransporterAgent,

         ###################################
         ### Single-Stream Architectures ###
         # Transporter-only
         'transporter': OriginalTransporterAgent,

         # CLIP-only without language
         'clip_unet_transporter': ClipUNetTransporterAgent,

         # CLIP-only
         'clip_lingunet_transporter': ClipLingUNetTransporterAgent,

         # CLIP-only with less layers
         'clip_lingunet_transporter_less_layers': ClipLingUNetTransporterAgentlesslayers,

         # Transporter with language (at bottleneck)
         'transporter_lang': OriginalTransporterLangFusionAgent,

         # Image-Goal Transporter
         'image_goal_transporter': ImageGoalTransporterAgent,

         ##############################################
         ### New variants NOT reported in the paper ###

         # CLIPort with FiLM language fusion
         'two_stream_clip_film_lingunet_lat_transporter': TwoStreamClipFilmLingUNetLatTransporterAgent,

         # CLIPort with CoCoOp
         'cocoop': TwoStreamClipLingUNetLatTransporterAgentCoCoOp,
         'coop': TwoStreamClipLingUNetLatTransporterAgentCoOp,
         'coop_rn50': TwoStreamClipLingUNetLatTransporterAgentCoOpRN50,
         'cocoop_rn50': TwoStreamClipLingUNetLatTransporterAgentCoCoOpRN50,

        #  CLIPort without decoder
        'conceptfusion_sam2': ConceptFusionSam2,
        'conceptfusion': ConceptFusionAgent,
        'conceptfusion_large': ConceptFusionLarge,
        'sam2clip': Sam2Clip,
        'sam2clip_wo': Sam2Clip_wo,

        # CLIP + TransporterNet + Sam2
        'cliport_sam2': TransFusionSam2,

        # CLIP with Unetr
        'unetr': UnetrAgent,

        # CLIPFit + Conceptfuion
        'clipfit': CLIPFitAgent,
        'clipfit1': CLIPFitAgent1,
        'clipfit_allsim': CLIPFitAllSimAgent,
        'pretrain': PretrainAgent,
        'real': CLIPFitRealAgent,
        }
