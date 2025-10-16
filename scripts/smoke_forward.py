import torch
import sys
import os
import copy
import argparse
import inspect

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.iTransformer_PS import Model

class Config:
    def __init__(self):
        # Basic configs
        self.seq_len = 96
        self.pred_len = 96
        self.output_attention = False
        self.use_norm = True
        self.d_model = 512
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.class_strategy = 'projection'
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 2048
        self.activation = 'gelu'
        self.factor = 1
        
        # PS module configs
        self.use_star = 0
        self.star_alpha = 0.7
        self.star_fuse = 'gate'
        self.star_gate_init = 0.5
        self.use_patch_embed = 0
        self.patch_len = 16
        self.patch_stride = 8
        self.patch_norm = 'instance'
        self.patch_pos_emb = 'sincos'
        self.use_ci = 1
        
        # 添加缺失的属性
        self.c_out = 7  # 对应输入通道数
        self.moving_avg = 25

def run_case(use_star, use_patch):
    B, L, C = 4, 96, 7
    x = torch.randn(B, L, C)
    
    config = Config()
    config.use_star = use_star
    config.use_patch_embed = use_patch
    
    model = Model(config)
    
    with torch.no_grad():
        y = model(x, None, None, None)
    
    print(f"case star={use_star} patch={use_patch} -> out.shape={tuple(y.shape)}")

if __name__ == "__main__":
    print("Running smoke tests for iTransformer_PS...")
    
    for s in (0, 1):
        for p in (0, 1):
            run_case(s, p)
    
    print("[OK] All smoke tests passed!")

    # ===== Patch-Modules 自检 =====
    print("\n[Self-Check] 开始模块与接口自检...")
    ok = True

    # 1) 必要模块是否可导入
    try:
        from modules.ms_patch import MSPatchEmbed
        from modules.ms_router import MSPatchRouter
        from modules.patch_filter import PatchFilter
        from modules.patch_mixer import PatchMixer
        print("[Self-Check] 模块导入 OK")
    except Exception as e:
        print("[Self-Check][ERROR] 模块导入失败：", e); ok = False

    # 2) Encoder / Attention 是否支持 attn_mask
    try:
        from layers.Transformer_EncDec import Encoder, EncoderLayer
        enc_has = "attn_mask" in inspect.signature(Encoder.forward).parameters
        encl_has = "attn_mask" in inspect.signature(EncoderLayer.forward).parameters
        if enc_has and encl_has:
            print("[Self-Check] Encoder 支持 attn_mask")
        else:
            print("[Self-Check][WARN] Encoder 未检测到 attn_mask 参数，请确认已按指引贯通传递")
    except Exception as e:
        print("[Self-Check][WARN] 无法自动检查 Encoder attn_mask：", e)

    # 3) Model 是否暴露 get_aux_loss（路由预算正则）
    try:
        from model.iTransformer_PS import Model as PSModel
        # 创建一个简单的配置用于测试
        test_config = argparse.Namespace(
            seq_len=96, pred_len=96, output_attention=False, use_norm=True,
            d_model=512, embed='timeF', freq='h', dropout=0.1, class_strategy='projection',
            n_heads=8, e_layers=2, d_ff=2048, activation='gelu', factor=1,
            use_star=0, use_patch_embed=0, use_ms_patch=0, use_ms_router=0,
            use_patch_filter=0, use_patch_mixer=0, c_out=7
        )
        m = PSModel(test_config)
        has_aux = hasattr(m, "get_aux_loss")
        print(f"[Self-Check] Model.get_aux_loss 存在={has_aux}")
    except Exception as e:
        print("[Self-Check][WARN] get_aux_loss 自检失败（可能构造模型需要完整 args）：", e)

    print("[Self-Check] 完成。\n")

    # ===== 四方案前向冒烟 =====
    print("Running patch modules smoke for iTransformer_PS...")

    def apply_overrides(a, kv: dict):
        for k, v in kv.items():
            setattr(a, k, v)
        return a

    CASES = [
        ("baseline",      dict(use_patch_embed=0, use_ms_patch=0, use_ms_router=0, use_patch_filter=0, use_patch_mixer=0)),

        ("ms_patch",      dict(use_ms_patch=1, ms_patch_set="8,16,32", ms_stride_ratio=0.5, ms_fuse="gate",
                               use_patch_embed=0, use_ms_router=0, use_patch_filter=0, use_patch_mixer=0)),

        ("ms_router",     dict(use_ms_router=1, router_hidden=64, router_budget=0.01,
                               use_ms_patch=0, use_patch_embed=0, use_patch_filter=0, use_patch_mixer=0)),

        ("patch_filter",  dict(use_patch_embed=1, patch_len=24, patch_stride=12, patch_norm="instance", patch_pos_emb="none",
                               use_patch_filter=1, filter_topk=8, filter_season="24,168",
                               use_ms_patch=0, use_ms_router=0, use_patch_mixer=0)),

        ("patch_mixer",   dict(use_patch_embed=1, patch_len=24, patch_stride=12, patch_norm="instance", patch_pos_emb="none",
                               use_patch_mixer=1, mixer_groups=8, mixer_dropout=0.1,
                               use_ms_patch=0, use_ms_router=0, use_patch_filter=0)),
    ]

    # 基本维度（与你之前 smoke 使用的保持一致）
    B, L, C = 4, 96, 321
    pred_len = 96

    # 准备一个基础 args（请沿用你原脚本已有的 args 构造；下面仅示例关键字段）
    base_args = argparse.Namespace(
        is_training=0, model_id="SMOKE", model="iTransformer_PS",
        data="custom", features="M", seq_len=L, label_len=48, pred_len=pred_len,
        enc_in=C, dec_in=C, c_out=C, d_model=512, d_ff=512, e_layers=3, n_heads=8,
        des="Exp", batch_size=4, learning_rate=5e-4, itr=1, use_gpu=False,
        # 默认关闭所有新增模块
        use_patch_embed=0, use_ms_patch=0, use_ms_router=0, use_patch_filter=0, use_patch_mixer=0,
        patch_len=16, patch_stride=8, patch_norm="instance", patch_pos_emb="none",
        ms_patch_set="8,16,32", ms_stride_ratio=0.5, ms_fuse="gate",
        router_hidden=64, router_budget=0.0,
        filter_topk=8, filter_season="",
        mixer_groups=8, mixer_dropout=0.1,
        use_star=0, use_ci=1,
        # 添加缺失的属性
        moving_avg=25,
        # 添加模型必需属性
        output_attention=False,
        use_norm=True,
        embed='timeF',
        freq='h',
        dropout=0.1,
        class_strategy='projection',
        activation='gelu',
        factor=1,
        # 添加其他可能需要的属性
        distil=True,
        use_amp=False,
        do_predict=False,
        inverse=False,
        channel_independence=False,
        efficient_training=False,
        partial_start_index=0
    )

    # 构造一组随机输入（与原 smoke 一致地准备 x_dec / x_mark）
    x_enc = torch.randn(B, L, C)
    x_mark_enc = torch.zeros(B, L, 4)
    dec_len = base_args.label_len + base_args.pred_len
    x_dec = torch.zeros(B, dec_len, C)
    x_mark_dec = torch.zeros(B, dec_len, 4)

    from model.iTransformer_PS import Model as PSModel

    ok_all = True
    for name, overrides in CASES:
        print(f"[CASE] {name} -> ", end="")
        a = apply_overrides(copy.deepcopy(base_args), overrides)
        try:
            m = PSModel(a)
            m.eval()
            with torch.no_grad():
                y = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
            assert y.shape == (B, pred_len, C), f"输出形状不符：{y.shape}"
            print(f"out.shape={tuple(y.shape)} ✔")
        except Exception as e:
            ok_all = False
            print("✘ ERROR:", e)

    print("[OK] Patch modules smoke passed!" if ok_all else "[FAIL] Patch modules smoke found errors.")