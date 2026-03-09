# OpenPI + 3D Gaussian Splatting 完整架构文档

> 📅 更新时间: 2026-03-07  
> 📝 作者: Zijian Zhang  
> 🎯 目标: 详细说明 PI0 + 3DGS 世界模型的完整架构

---

## 📋 目录

1. [整体架构概览](#整体架构概览)
2. [模块 1: ENCODER (多模态特征提取)](#模块-1-encoder-多模态特征提取)
3. [模块 2: VLM (视觉-语言融合)](#模块-2-vlm-视觉-语言融合)
4. [模块 3a: DECODER (World Model)](#模块-3a-decoder-world-model)
5. [模块 4: RENDER (3D场景渲染)](#模块-4-render-3d场景渲染)
6. [模块 3b: ACTION EXPERT (动作预测)](#模块-3b-action-expert-动作预测)
7. [完整训练流程](#完整训练流程)
8. [关键参数总结](#关键参数总结)
9. [常见问题 FAQ](#常见问题-faq)

---

## 🏗️ 整体架构概览

### 数据流图

```
输入数据 (DROID/ALOHA Dataset)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. ENCODER: 多模态特征提取                                    │
│    ├─ Vision Encoder (SigLIP)                                │
│    ├─ Language Encoder (Gemma Embedding)                     │
│    └─ 3DGS Encoder (GaussianAdapter/VGGT)                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. VLM: 视觉-语言融合 (PaliGemma Transformer)                │
│    - 多模态上下文理解                                         │
│    - 输出统一的语义特征                                       │
└─────────────────────────────────────────────────────────────┘
    ↓
    ├─→ 3a. DECODER (World Model): 未来场景预测
    │       ↓
    │   ┌─────────────────────────────────────────────────┐
    │   │ GaussianDecoder: 3DGS参数生成                   │
    │   │  - 多尺度特征融合 (DPT-style)                    │
    │   │  - 输出: rotation, scale, opacity, SH           │
    │   └─────────────────────────────────────────────────┘
    │       ↓
    │   ┌─────────────────────────────────────────────────┐
    │   │ 4. RENDER: 3D场景渲染                           │
    │   │  - depth2pc: 深度→3D点云                        │
    │   │  - GaussianRenderer: 光栅化                     │
    │   │  - 输出: 预测的未来帧图像                        │
    │   └─────────────────────────────────────────────────┘
    │
    └─→ 3b. ACTION EXPERT: 动作预测
        ↓
    ┌─────────────────────────────────────────────────────┐
    │ Gemma Expert Transformer                             │
    │  - Flow Matching解码                                 │
    │  - 输出: 机器人动作序列                              │
    └─────────────────────────────────────────────────────┘
```

### 核心设计理念

| 设计原则 | 说明 |
|---------|------|
| **解耦设计** | VLM专注语义理解，Decoder专注几何生成 |
| **模块化** | 每个模块可独立训练和优化 |
| **多任务学习** | 同时学习动作预测和场景预测 |
| **端到端训练** | 渲染损失直接监督世界模型 |

---

## 🔧 模块 1: ENCODER (多模态特征提取)

### 1.1 Vision Encoder (SigLIP)

**📁 文件位置**: `src/openpi/models_pytorch/gemma_pytorch.py` → `PaliGemmaForConditionalGeneration.vision_tower`

**🎯 功能**: 将输入图像编码为视觉特征

#### 输入数据

```python
agent_view: [B, T, 224, 224, 3]  # T=3 (历史帧: t-2, t-1, t)
wrist_view: [B, T, 224, 224, 3]  # 手腕相机视角
```

#### 处理流程

```
Step 1: Patch Embedding
  图像 [224×224] → 14×14 patches → [16×16 tokens]

Step 2: Vision Transformer (SigLIP)
  - 架构: 12层 Transformer
  - 注意力头: 16 heads
  - 隐藏维度: 1152
  输出: [B, T, 256, 1152]  # 256 = 16×16 patches

Step 3: Multi-modal Projector
  - Linear层: 1152 → 2048
  - 激活函数: GELU
  输出: [B, T, 256, 2048]

Step 4: 展平时序维度
  [B, T, 256, 2048] → [B, T×256, 2048]
  # T=3 时, 总共 768 个vision tokens
```

#### 关键代码

```python
# pi0_pytorch.py:594-620
img_emb = self.paligemma_with_expert.embed_images(img)
# img_emb: [B, T, 256, 2048]

# 展平时序维度
if has_temporal:
    B, T, N, D = img_emb.shape
    img_emb = img_emb.view(B, T * N, D)  # [B, T*256, 2048]
```

#### 参数量

- SigLIP Encoder: ~400M 参数
- 状态: **冻结** (不参与训练)

---

### 1.2 Language Encoder (Gemma Embedding)

**📁 文件位置**: `src/openpi/models_pytorch/gemma_pytorch.py` → `PaliGemmaForConditionalGeneration.language_model.embed_tokens`

**🎯 功能**: 将文本指令编码为语言特征

#### 输入数据

```python
tokenized_prompt: [B, L]  # L = 序列长度
# 示例: "pick up the red cup" → [257001, 12345, 67890, ...]
```

#### 处理流程

```
Step 1: Token Embedding
  - Vocabulary Size: 257,152
  - Embedding Dimension: 2048
  - 查表: token_id → embedding vector

Step 2: Scaling (稳定训练)
  emb = emb * sqrt(2048) ≈ emb * 45.25

  原因: 匹配 Vision Encoder 的输出尺度
```

#### 关键代码

```python
# pi0_pytorch.py:526-531
lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
lang_emb_dim = lang_emb.shape[-1]
lang_emb = lang_emb * math.sqrt(lang_emb_dim)  # Scaling
```

#### 输出

```python
lang_emb: [B, L, 2048]
lang_mask: [B, L]  # 1=有效token, 0=padding
```

#### 参数量

- Embedding Table: ~500M 参数
- 状态: **冻结** (不参与训练)

---

### 1.3 3DGS Encoder (GaussianAdapter/VGGT)

**📁 文件位置**: `src/openpi/models_pytorch/pi0_vggt.py` → `GaussianAdapter`

**🎯 功能**: 从历史帧中提取时序3D特征，生成future query tokens

#### 输入数据

```python
images: [B, T, 224, 224, 3]        # T=3 历史帧
depth_maps: [B, T, H, W, 1]        # 深度图 (可选)
camera_params: Dict {              # 相机参数
    "intrinsics": [B, T, 3, 3],
    "extrinsics": [B, T, 4, 4]
}
text_embedding: [B, D]             # 文本特征 (用于LGPD)
```

#### 架构选项

##### Option 1: TemporalConv3D

```python
输入: [B, C, T, H, W]
  ↓
Conv3D(3×3×3, stride=(1,2,2))  # 保留时序, 下采样空间
  ↓
GroupNorm + GELU
  ↓
Conv3D(3×3×3, stride=(1,2,2))
  ↓
输出: [B, 512, T, H/4, W/4]
```

##### Option 2: CausalTemporalAttention

```python
# 因果注意力: 每帧只能attend到自己和之前的帧
Attention Mask:
  t-2: [1, 0, 0]  # 只能看自己
  t-1: [1, 1, 0]  # 可以看 t-2 和自己
  t:   [1, 1, 1]  # 可以看所有历史
```

##### Option 3: LoRA微调

```python
# 冻结VGGT主干, 只训练轻量级适配器
for layer in vggt_encoder.layers:
    layer.requires_grad = False

# 添加LoRA
apply_lora_to_model(
    vggt_encoder,
    target_names=["qkv", "proj"],
    rank=8,
    alpha=32.0
)
```

#### 处理流程

```
Step 1: VGGT Encoder (冻结)
  输入: [B, T, 224, 224, 3]
    ↓
  Vision Transformer (预训练)
    ↓
  输出: [B, T, 37, 37, 2048]  # 37×37 = 1369 tokens per frame

Step 2: Temporal Modeling
  方案A: 3D Conv
    [B, T, 37, 37, 2048] → [B, 512, T, 9, 9]

  方案B: Temporal Attention
    [B, T×1369, 2048] → [B, T×1369, 2048]
    # 带因果掩码

Step 3: Future Query Generation
  # 聚合时序信息, 生成future tokens
  [B, T, 37, 37, 2048] → [B, 256, 2048]

  方法:
    - Pooling (spatial + temporal)
    - Learnable Query Tokens
    - Cross-Attention
```

#### 关键代码

```python
# pi0_pytorch.py:584
gaussian_embs, g_mask = self.gaussian_adapter(
    gaussian_inputs,
    text_embedding=text_embedding
)

# pi0_vggt.py:200-300 (GaussianAdapter.forward)
# 1. VGGT编码
vggt_features = self.vggt_encoder(images)  # [B, T, 37, 37, 2048]

# 2. 时序建模
if self.use_temporal_conv:
    temporal_features = self.temporal_conv3d(vggt_features)
elif self.use_temporal_attention:
    temporal_features = self.causal_attention(vggt_features)

# 3. 生成future tokens
future_tokens = self.future_query_generator(temporal_features)
# [B, 256, 2048]
```

#### 输出

```python
future_tokens: [B, 256, 2048]  # 256个查询token
gaussian_mask: [B, 256]        # 全1 (所有token有效)
```

#### 参数量

- VGGT Encoder: ~1B 参数 (冻结)
- LoRA Adapter: ~50M 参数 (可训练)
- Temporal Modules: ~20M 参数 (可训练)

---

## 🔗 模块 2: VLM (视觉-语言融合)

**📁 文件位置**: `src/openpi/models_pytorch/gemma_pytorch.py` → `PaliGemmaForConditionalGeneration`

**🎯 功能**: 融合多模态特征，生成统一的语义表示

### 输入 (Prefix Tokens)

```python
1. Gaussian Tokens: [B, 256, 2048]      # 3DGS特征
2. Vision Tokens:   [B, T×256, 2048]    # 图像特征 (T=3, 共768 tokens)
3. Language Tokens: [B, L, 2048]        # 文本指令

拼接顺序:
  [Gaussian | Vision | Language]
  总长度: 256 + 768 + L ≈ 1024 + L
```

### 处理流程

```
Step 1: Token拼接
  embs = torch.cat([gaussian_embs, vision_embs, lang_emb], dim=1)
  # [B, N, 2048] where N = 256 + 768 + L

Step 2: Attention Mask构建
  Gaussian Tokens: att_mask = 0 (context, 可被后续attend)
  Vision Tokens:   att_mask = 0 (context)
  Language Tokens: att_mask = 0 (context)
  
  # 2D Attention Mask (因果掩码)
  att_2d_mask[i, j] = (cumsum[i] <= cumsum[j]) & pad_mask[i] & pad_mask[j]

Step 3: PaliGemma Transformer (18层)
  for layer in layers:
    # Multi-Head Self-Attention
    x = LayerNorm(x)
    x = x + MultiHeadAttention(x, mask=att_2d_mask)
    
    # Feed-Forward Network
    x = LayerNorm(x)
    x = x + FFN(x)
  
  输出: [B, N, 2048]

Step 4: 输出分段
  - Gaussian部分: prefix_output[:, :256, :]
  - Vision部分:   prefix_output[:, 256:1024, :]
  - Language部分: prefix_output[:, 1024:, :]
```

### 关键代码

```python
# pi0_pytorch.py:506-650 (embed_prefix)
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, gaussian_inputs):
    embs = []
    pad_masks = []
    att_masks = []
    
    # 1. Gaussian Tokens
    gaussian_embs, g_mask = self.gaussian_adapter(gaussian_inputs)
    embs.append(gaussian_embs)
    pad_masks.append(g_mask)
    att_masks += [0] * 256  # Context tokens
    
    # 2. Vision Tokens
    for img, img_mask in zip(images, img_masks):
        img_emb = self.paligemma_with_expert.embed_images(img)
        embs.append(img_emb)
        pad_masks.append(img_mask)
        att_masks += [0] * img_emb.shape[1]
    
    # 3. Language Tokens
    lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks += [0] * lang_emb.shape[1]
    
    # 4. 拼接
    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks).expand(B, -1)
    
    # 5. 构建2D mask
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    
    # 6. PaliGemma Transformer
    prefix_output = self.paligemma_with_expert.paligemma.language_model(
        inputs_embeds=embs,
        attention_mask=att_2d_masks
    ).last_hidden_state
    
    return prefix_output
```

### 输出

```python
vlm_features: [B, N, 2048]  # N = 总token数
segment_lengths: Dict {
    'gaussian': 256,
    'images': 768,
    'language': L
}
```

### 参数量

- PaliGemma Transformer: ~3B 参数
- 状态: **冻结** (不参与训练)

---

## 🎨 模块 3a: DECODER (World Model - 未来场景预测)

### GaussianDecoder (3DGS参数生成)

**📁 文件位置**: `src/openpi/models_pytorch/pi0_world_model.py` → `GaussianDecoder`

**🎯 功能**: 将future tokens解码为密集的3D Gaussian参数图

#### 输入数据

```python
future_tokens: [B, 256, 2048]  # 从VLM提取的future tokens
depth_maps: [B, H, W, 1]       # 当前帧深度图 (用于初始化)
```

#### 架构: DPT-style多尺度上采样

```
16×16 (2048ch) → 32×32 (512ch) → 64×64 (256ch) 
    → 128×128 (128ch) → 256×256 (17ch)

每层包含:
  - ConvTranspose2d (上采样 2x)
  - GroupNorm + GELU
  - Residual Connection
  - Feature Fusion Block (DPT-style)
```

#### 处理流程

```
Step 1: Reshape tokens
  [B, 256, 2048] → [B, 2048, 16, 16]

Step 2: 多尺度上采样
  layer1: 16×16 → 32×32 (512ch)
    h1 = ConvTranspose2d(z, 512, kernel=4, stride=2)
    h1 = GroupNorm(h1) + GELU
    h1 = h1 + upsample(skip_connection)
  
  layer2: 32×32 → 64×64 (256ch)
  layer3: 64×64 → 128×128 (128ch)
  layer4: 128×128 → 256×256 (64ch)

Step 3: 输出头 (17通道)
  raw = Conv2d(h4, 17, kernel=3, padding=1)
  
  分割通道:
    - rotation:  raw[:, 0:4, :, :]   # 4通道
    - scale:     raw[:, 4:7, :, :]   # 3通道
    - opacity:   raw[:, 7:8, :, :]   # 1通道
    - SH:        raw[:, 8:17, :, :]  # 9通道

Step 4: 激活函数
  # Rotation: 归一化四元数
  rot_maps = rot_raw / (rot_raw.norm(dim=-1, keepdim=True) + 1e-8)
  
  # Scale: softplus + 缩放
  scale_maps = F.softplus(scale_raw) * 0.01  # ✅ 修复后 (原0.001)
  
  # Opacity: sigmoid
  opacity_maps = torch.sigmoid(opa_raw)
  
  # SH: 带mask衰减
  sh_maps = sh_raw * sh_mask  # 衰减高阶系数

Step 5: Depth插值
  # 将depth从VGGT分辨率插值到decoder分辨率
  depth_maps = F.interpolate(depth_maps, size=(256, 256), mode='bilinear')
```

#### 关键代码

```python
# pi0_world_model.py:281-380 (GaussianDecoder.decode)
def decode(self, z, future_observation=None, depth_maps=None, frame_idx=-1):
    B = z.shape[0]
    
    # 1. Reshape
    z_spatial = z.view(B, self.grid_size, self.grid_size, -1)
    z_spatial = z_spatial.permute(0, 3, 1, 2)  # [B, 2048, 16, 16]
    
    # 2. 多尺度上采样
    h1 = self.gaussian_head.layer1(z_spatial)  # 32×32
    h2 = self.gaussian_head.layer2(h1)         # 64×64
    h3 = self.gaussian_head.layer3(h2)         # 128×128
    h4 = self.gaussian_head.layer4(h3)         # 256×256
    
    # 3. 输出头
    raw = self.gaussian_head.output_conv(h4)   # [B, 17, 256, 256]
    rot_raw, scale_raw, opa_raw, sh_raw = torch.split(raw, [4, 3, 1, 9], dim=1)
    
    # 4. 激活
    rot_maps = rot_raw.permute(0, 2, 3, 1)
    rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
    
    scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * 0.01
    
    opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))
    
    sh_maps = sh_raw.permute(0, 2, 3, 1)
    sh_mask = self.gaussian_head.sh_mask.view(1, 1, 1, 9)
    sh_maps = sh_maps * sh_mask
    
    # 5. Depth插值
    final_depth = F.interpolate(
        depth_maps[:, frame_idx, :, :, 0].unsqueeze(1),
        size=(256, 256),
        mode='bilinear'
    ).squeeze(1)
    
    return {
        "rot_maps": rot_maps,
        "scale_maps": scale_maps,
        "opacity_maps": opacity_maps,
        "sh_maps": sh_maps,
        "depth_maps": final_depth
    }
```

#### 输出

```python
gaussian_params: Dict {
    "rot_maps":     [B, 256, 256, 4],  # 四元数
    "scale_maps":   [B, 256, 256, 3],  # xyz尺度
    "opacity_maps": [B, 256, 256, 1],  # 不透明度
    "sh_maps":      [B, 256, 256, 9],  # 球谐系数
    "depth_maps":   [B, 256, 256]      # 深度图
}
```

#### 参数量

- GaussianDecoder: ~100M 参数
- 状态: **可训练** (主要训练目标)

#### 最近修复

```python
# ❌ 修改前: scale太小导致渲染质量差
scale_maps = F.softplus(scale_raw) * 0.001

# ✅ 修改后: 增大10倍
scale_maps = F.softplus(scale_raw) * 0.01

原因:
  - 0.001太小 → Gaussian覆盖像素少 → 渲染质量差
  - 0.01合理 → 符合3DGS标准范围 (0.01-1.0)
```

---

## 🎬 模块 4: RENDER (3D场景渲染)

**📁 文件位置**: `src/openpi/models_pytorch/gaussian_renderer.py` → `GaussianRenderer`

**🎯 功能**: 将3D Gaussian参数渲染为2D图像，用于监督训练

### 输入数据

```python
gaussian_params: Dict {
    "rot_maps":     [B, 256, 256, 4],
    "scale_maps":   [B, 256, 256, 3],
    "opacity_maps": [B, 256, 256, 1],
    "sh_maps":      [B, 256, 256, 9],
    "depth_maps":   [B, 256, 256]
}

camera_params: Dict {
    "viewmatrix":  [B, 4, 4],  # 相机外参 (Identity)
    "projmatrix":  [B, 4, 4],  # 投影矩阵
    "intrinsics":  [B, 3, 3],  # 相机内参
    "tanfovx":     float,      # FOV (x方向)
    "tanfovy":     float,      # FOV (y方向)
    "campos":      [B, 3]      # 相机位置 (0,0,0)
}
```

### 处理流程

#### Step 1: depth2pc (深度图 → 3D点云)

```python
输入: depth_maps [B, 256, 256]

# 相机空间坐标计算
u = torch.arange(0.5, W + 0.5)  # [W] 像素坐标 (带半像素偏移)
v = torch.arange(0.5, H + 0.5)  # [H]
v_grid, u_grid = torch.meshgrid(v, u)

# 反投影公式
x = (u_grid - cx) * depth / fx
y = (v_grid - cy) * depth / fy
z = depth

xyz = torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]
xyz = xyz.reshape(B, H * W, 3)        # [B, N, 3] where N=65536

输出: xyz [B, 65536, 3]  # 3D点云 (相机空间)
```

**为什么用单位矩阵作为viewmatrix?**

```
✅ 正确原因:
  - depth2pc已经输出相机空间坐标 (z = depth > 0)
  - 相机在原点 (0, 0, 0), 看向 +Z 方向
  - 不需要额外的 world-to-camera 变换
  - viewmatrix = Identity 是正确的设计

❌ 错误做法:
  - 添加平移 (如 z += 5.0) 会把Gaussians推远
  - 导致渲染异常 (全灰色或空白)
```

#### Step 2: 参数准备

```python
# Reshape参数图为点云格式
rotations = rot_maps.reshape(B, -1, 4)      # [B, 65536, 4]
scales = scale_maps.reshape(B, -1, 3)       # [B, 65536, 3]
opacity = opacity_maps.reshape(B, -1, 1)    # [B, 65536, 1]
shs = sh_maps.reshape(B, -1, 9)             # [B, 65536, 9]

gaussian_params = {
    "xyz": xyz,
    "rotations": rotations,
    "scales": scales,
    "opacity": opacity,
    "shs": shs
}
```

#### Step 3: 数值清理

```python
# NaN/Inf检测与替换
if torch.isnan(xyz).any() or torch.isinf(xyz).any():
    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=10.0, neginf=-10.0)

# Clamp scales
scales = torch.clamp(scales, min=1e-6, max=10.0)

# Clamp opacity
opacity = torch.clamp(opacity, min=1e-6, max=1.0)

# 归一化四元数
rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
```

#### Step 4: 相机变换

```python
# Viewmatrix (相机外参)
viewmatrix = torch.eye(4, device=device)  # Identity
viewmatrix = viewmatrix.unsqueeze(0).repeat(B, 1, 1)

# Projection Matrix (投影矩阵)
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0
    
    return P

projmatrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX, fovY)

# Full MVP Matrix
full_proj_transform = projmatrix @ viewmatrix  # [B, 4, 4]
```

#### Step 5: 3DGS光栅化 (AD-FFgsStudio)

```python
# 配置光栅化器
raster_settings = GaussianRasterizationSettings(
    image_height=224,
    image_width=224,
    tanfovx=math.tan(fovX / 2),
    tanfovy=math.tan(fovY / 2),
    bg=torch.tensor([0, 0, 0], device=device),  # 背景色
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=1,  # DC + 1阶球谐
    campos=torch.tensor([0, 0, 0], device=device),
    prefiltered=False,
    debug=False
)

rasterizer = GaussianRasterizer(raster_settings)

# 光栅化
rendered_image, radii = rasterizer(
    means3D=xyz,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=None,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None
)

# 输出: [B, 3, H, W]
```

**光栅化原理**:

```
1. 投影到2D屏幕空间
   p_screen = projmatrix @ viewmatrix @ p_3d

2. 深度排序
   按z值从近到远排序Gaussians

3. Alpha混合 (从前到后)
   C = Σᵢ αᵢ cᵢ Πⱼ<ᵢ (1-αⱼ)
   
   其中:
     αᵢ = opacity * exp(-0.5 * d²/σ²)
     cᵢ = SH_color(view_dir)
     d = 像素到Gaussian中心的距离

4. 球谐着色 (view-dependent)
   color = SH_DC + SH_1st_order(view_dir)
```

#### Step 6: 输出

```python
rendered_image: [B, 3, 224, 224]  # RGB图像
```

### 损失计算

```python
def compute_multi_view_rendering_loss(
    rendered_images,
    target_images,
    lambda_ssim=0.8,
    lambda_l1=0.2,
    lambda_scale=0.001,
    lambda_opacity=0.01
):
    # 1. 光度损失
    ssim_loss = 1.0 - ssim(rendered_images, target_images)
    l1_loss = F.l1_loss(rendered_images, target_images)
    photo_loss = lambda_ssim * ssim_loss + lambda_l1 * l1_loss
    
    # 2. 正则化
    scale_reg = gaussian_params["scales"].norm(dim=-1).mean()
    opacity_reg = gaussian_params["opacity"].mean()
    reg_loss = lambda_scale * scale_reg + lambda_opacity * opacity_reg
    
    # 3. 总损失
    total_loss = photo_loss + reg_loss
    
    return total_loss
```

### 关键代码

```python
# gaussian_renderer.py:241-400 (GaussianRenderer.render)
def render(self, gaussian_params, camera_params, target_image=None, step=None):
    # 1. depth2pc
    xyz = self.depth2pc(
        gaussian_params["depth_maps"],
        fx, fy, cx, cy
    )  # [B, N, 3]
    
    # 2. 参数准备
    rotations = gaussian_params["rot_maps"].reshape(B, -1, 4)
    scales = gaussian_params["scale_maps"].reshape(B, -1, 3)
    opacity = gaussian_params["opacity_maps"].reshape(B, -1, 1)
    shs = gaussian_params["sh_maps"].reshape(B, -1, 9)
    
    # 3. 数值清理
    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=10.0, neginf=-10.0)
    scales = torch.clamp(scales, min=1e-6, max=10.0)
    opacity = torch.clamp(opacity, min=1e-6, max=1.0)
    rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 4. 光栅化
    rendered_image = self._rasterize(
        xyz, rotations, scales, opacity, shs,
        camera_params
    )  # [B, 3, H, W]
    
    # 5. 损失计算
    if target_image is not None:
        loss = compute_multi_view_rendering_loss(
            rendered_image, target_image
        )
        return rendered_image, loss
    
    return rendered_image
```

### 参数量

- GaussianRenderer: **0参数** (纯算法, 无可学习参数)
- 依赖: AD-FFgsStudio CUDA光栅化器

---

## 🤖 模块 3b: ACTION EXPERT (动作预测)

**📁 文件位置**: `src/openpi/models_pytorch/gemma_pytorch.py` → `GemmaForCausalLM` (action_expert)

**🎯 功能**: 从VLM特征中预测机器人动作序列

### 输入数据

```python
1. Prefix Features (来自VLM):
   prefix_output: [B, N, 2048]  # N = Gaussian + Vision + Language

2. Noisy Actions (Flow Matching):
   noisy_actions: [B, H, 32]  # H = action_horizon (如16步)
   # 加入高斯噪声: a_t = a_0 + t * ε

3. Timestep:
   t: [B]  # t ∈ [0, 1], Flow matching时间步
```

### Flow Matching原理

```
训练目标: 学习从噪声到真实动作的速度场

1. 前向过程 (加噪):
   a_t = (1-t) * a_0 + t * a_1 + σ_t * ε
   
   其中:
     a_0 = 随机噪声 ~ N(0, I)
     a_1 = 真实动作
     t ~ Beta(1.5, 1.0)  # 时间步采样
     ε ~ N(0, I)         # 高斯噪声

2. 速度场 (velocity):
   v_t = (a_1 - a_t) / t
   
   物理意义: 从当前状态a_t到目标a_1的"速度"

3. 训练损失:
   L = MSE(model(a_t, t), v_t)
   
   模型学习预测速度场

4. 推理 (ODE求解):
   a_0 = randn(B, H, 32)
   for t in [0, 0.1, 0.2, ..., 1.0]:
     v_t = model(a_t, t)
     a_{t+dt} = a_t + v_t * dt
   return a_1  # 最终动作
```

### 处理流程

#### Step 1: Action Embedding

```python
# 投影到expert维度
action_emb = self.action_in_proj(noisy_actions)  # [B, H, 32] → [B, H, 2048]
```

#### Step 2: Timestep Embedding

```python
# Sinusoidal位置编码
def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period):
    fraction = torch.linspace(0.0, 1.0, dimension // 2)
    period = min_period * (max_period / min_period) ** fraction
    
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

time_emb = create_sinusoidal_pos_embedding(
    t, 
    dim=2048, 
    min_period=4e-3, 
    max_period=4.0
)  # [B, 2048]
```

#### Step 3: 融合时间信息

```python
if pi05:
    # AdaRMS: 自适应归一化
    adarms_cond = self.time_mlp_in(time_emb)
    adarms_cond = F.silu(adarms_cond)
    adarms_cond = self.time_mlp_out(adarms_cond)
    adarms_cond = F.silu(adarms_cond)  # [B, 2048]
    
    action_time_emb = action_emb  # [B, H, 2048]
    
else:
    # 拼接融合
    time_emb = time_emb[:, None, :].expand(-1, H, -1)  # [B, H, 2048]
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # [B, H, 4096]
    
    # MLP融合
    action_time_emb = self.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = self.action_time_mlp_out(action_time_emb)  # [B, H, 2048]
```

#### Step 4: Attention Mask

```python
# 防止action tokens attend到prefix
# 只有第1个action可以attend所有prefix, 后续action只能attend自己

if pi05:
    # State不参与 (pi05没有state)
    att_masks = [1] + [0] * (H - 1)
else:
    # State参与
    att_masks = [1] + [1] + [0] * (H - 1)
    #          state  action[0]  action[1:]

# 构建2D mask
att_2d_mask = make_att_2d_masks(pad_masks, att_masks)
```

#### Step 5: Gemma Expert Transformer (14层)

```python
# 输入: [prefix_features | action_time_emb]
full_input = torch.cat([prefix_output, action_time_emb], dim=1)
# [B, N+H, 2048]

expert_output = self.gemma_expert.model(
    inputs_embeds=full_input,
    attention_mask=att_2d_mask,
    adarms_cond=adarms_cond  # pi05专用
).last_hidden_state  # [B, N+H, 2048]
```

**AdaRMS (Adaptive RMS Normalization)**:

```python
# pi05专用: 用时间步条件调制归一化
class AdaRMSNorm(nn.Module):
    def forward(self, x, cond):
        # RMS Normalization
        x_norm = x / (x.pow(2).mean(-1, keepdim=True).sqrt() + eps)
        
        # 条件调制
        scale = 1.0 + self.scale_mlp(cond)
        shift = self.shift_mlp(cond)
        
        return x_norm * scale + shift

# 在每个Transformer层中:
x = AdaRMSNorm(x, adarms_cond)
x = x + Attention(x)
x = AdaRMSNorm(x, adarms_cond)
x = x + FFN(x)
```

#### Step 6: Action Prediction

```python
# 提取action部分的输出
action_output = expert_output[:, -H:, :]  # [B, H, 2048]

# 投影到action空间
pred_actions = self.action_out_proj(action_output)  # [B, H, 32]
```

### 训练

```python
# 1. 采样timestep
t = sample_beta(1.5, 1.0, B)  # Beta分布采样
t = t * 0.999 + 0.001  # 避免t=0或t=1

# 2. 加噪声
noisy_actions = clean_actions + t[:, None, None] * randn_like(clean_actions)

# 3. 计算速度目标
velocity_target = (clean_actions - noisy_actions) / t[:, None, None]

# 4. 前向传播
pred_velocity = model(
    observation,
    noisy_actions,
    t
)

# 5. 损失
action_loss = F.mse_loss(pred_velocity, velocity_target)
```

### 推理

```python
# ODE求解 (Euler方法)
def sample_actions(model, observation, num_steps=10):
    B, H, D = batch_size, action_horizon, action_dim
    
    # 初始化: 纯噪声
    actions = torch.randn(B, H, D, device=device)
    
    # 逐步去噪
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        
        # 预测速度
        velocity = model(observation, actions, t)
        
        # 更新动作
        actions = actions + velocity * dt
    
    return actions
```

### 关键代码

```python
# pi0_pytorch.py:700-776 (embed_action_suffix)
def embed_action_suffix(self, state, noisy_actions, timestep):
    embs = []
    pad_masks = []
    att_masks = []
    
    if not self.pi05:
        # State embedding
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        pad_masks.append(torch.ones(B, 1, dtype=torch.bool))
        att_masks += [1]
    
    # Timestep embedding
    time_emb = create_sinusoidal_pos_embedding(timestep, ...)
    
    # Action embedding
    action_emb = self.action_in_proj(noisy_actions)
    
    # 融合
    if self.pi05:
        adarms_cond = self.time_mlp(time_emb)
        action_time_emb = action_emb
    else:
        action_time_emb = self.action_time_mlp([action_emb, time_emb])
    
    embs.append(action_time_emb)
    pad_masks.append(torch.ones(B, H, dtype=torch.bool))
    att_masks += [1] + [0] * (H - 1)
    
    return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1), att_masks, adarms_cond

# pi0_pytorch.py:900-1000 (forward)
def forward(self, observation, noisy_actions, timestep):
    # 1. Prefix
    prefix_output = self.embed_prefix(
        observation.images,
        observation.image_masks,
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        gaussian_inputs
    )
    
    # 2. Action suffix
    action_suffix, action_mask, att_masks, adarms_cond = self.embed_action_suffix(
        observation.state,
        noisy_actions,
        timestep
    )
    
    # 3. 拼接
    full_input = torch.cat([prefix_output, action_suffix], dim=1)
    full_mask = torch.cat([prefix_mask, action_mask], dim=1)
    att_2d_mask = make_att_2d_masks(full_mask, att_masks)
    
    # 4. Gemma Expert
    expert_output = self.paligemma_with_expert.gemma_expert.model(
        inputs_embeds=full_input,
        attention_mask=att_2d_mask,
        adarms_cond=adarms_cond
    ).last_hidden_state
    
    # 5. Action prediction
    action_output = expert_output[:, -H:, :]
    pred_actions = self.action_out_proj(action_output)
    
    return pred_actions
```

### 输出

```python
pred_actions: [B, H, 32]  # H=16步动作序列

动作维度 (32D):
  - 左臂关节: 7D (joint positions)
  - 右臂关节: 7D
  - 左手夹爪: 1D (gripper)
  - 右手夹爪: 1D
  - 其他: 16D (根据机器人平台不同)
```

### 参数量

- Gemma Expert: ~1.5B 参数
- 状态: **可训练** (主要训练目标)

---

## 🔄 完整训练流程

### 训练伪代码

```python
# 初始化模型
model = PI0Pytorch(config)

# 冻结预训练模块
model.paligemma_with_expert.requires_grad_(False)  # 冻结VLM
model.gaussian_adapter.vggt_encoder.requires_grad_(False)  # 冻结VGGT

# 只训练
# - GaussianDecoder (世界模型)
# - GaussianAdapter的LoRA (时序建模)
# - Action Expert (动作预测)

optimizer = torch.optim.AdamW([
    {'params': model.gaussian_decoder.parameters(), 'lr': 1e-4},
    {'params': model.gaussian_adapter.lora_params(), 'lr': 1e-4},
    {'params': model.paligemma_with_expert.gemma_expert.parameters(), 'lr': 1e-4}
])

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # ========== 数据准备 ==========
        observation = batch['observation']  # 包含images, prompts, state
        actions = batch['actions']          # [B, H, 32]
        future_image = batch['future_image']  # [B, 3, 224, 224]
        
        # ========== 1. ENCODER ==========
        # 1.1 Vision Encoder (冻结)
        vision_tokens = model.paligemma_with_expert.embed_images(
            observation.images
        )  # [B, T×256, 2048]
        
        # 1.2 Language Encoder (冻结)
        lang_tokens = model.paligemma_with_expert.embed_language_tokens(
            observation.tokenized_prompt
        )  # [B, L, 2048]
        
        # 1.3 3DGS Encoder (LoRA可训练)
        gaussian_inputs = model._prepare_gaussian_inputs(observation)
        future_tokens, g_mask = model.gaussian_adapter(
            gaussian_inputs
        )  # [B, 256, 2048]
        
        # ========== 2. VLM (冻结) ==========
        prefix_features = model.embed_prefix(
            observation.images,
            observation.image_masks,
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            gaussian_inputs
        )  # [B, N, 2048]
        
        # ========== 3a. DECODER (可训练) ==========
        # 提取future tokens (从prefix中)
        future_tokens_from_prefix = prefix_features[:, :256, :]
        
        # GaussianDecoder
        gaussian_params = model.gaussian_decoder.decode(
            future_tokens_from_prefix,
            depth_maps=observation.depth_maps,
            frame_idx=-1  # 预测未来帧
        )  # Dict with rot, scale, opacity, SH, depth
        
        # ========== 4. RENDER (无参数) ==========
        camera_params = model._prepare_camera_params(observation)
        
        rendered_image, render_loss = model.gaussian_renderer.render(
            gaussian_params,
            camera_params,
            target_image=future_image,
            step=global_step
        )  # [B, 3, 224, 224]
        
        # ========== 3b. ACTION EXPERT (可训练) ==========
        # 采样timestep
        t = model.sample_time(B, device)  # [B]
        
        # 加噪声
        noise = torch.randn_like(actions)
        noisy_actions = actions + t[:, None, None] * noise
        
        # 计算速度目标
        velocity_target = (actions - noisy_actions) / t[:, None, None]
        
        # Action Expert
        action_suffix, action_mask, att_masks, adarms_cond = model.embed_action_suffix(
            observation.state,
            noisy_actions,
            t
        )
        
        full_input = torch.cat([prefix_features, action_suffix], dim=1)
        full_mask = torch.cat([prefix_mask, action_mask], dim=1)
        att_2d_mask = make_att_2d_masks(full_mask, att_masks)
        
        expert_output = model.paligemma_with_expert.gemma_expert.model(
            inputs_embeds=full_input,
            attention_mask=att_2d_mask,
            adarms_cond=adarms_cond
        ).last_hidden_state
        
        action_output = expert_output[:, -H:, :]
        pred_velocity = model.action_out_proj(action_output)
        
        # Flow matching损失
        action_loss = F.mse_loss(pred_velocity, velocity_target)
        
        # ========== 总损失 ==========
        render_loss_weight = 0.5  # 可调节
        total_loss = action_loss + render_loss_weight * render_loss
        
        # ========== 反向传播 ==========
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪 (防止NaN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # ========== 日志 ==========
        if global_step % 40 == 0:
            print(f"Step {global_step}")
            print(f"  Action Loss: {action_loss.item():.4f}")
            print(f"  Render Loss: {render_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
            
            # 可视化 (每100步)
            if global_step % 100 == 0:
                save_visualization(
                    rendered_image,
                    future_image,
                    save_dir=model.vis_save_dir,
                    step=global_step
                )
        
        global_step += 1
```

### 训练配置

```python
# src/openpi/training/config.py
config = {
    # 模型
    "use_gaussian": True,
    "use_world_model": True,
    "unfreeze_vggt_decoder_only": True,
    
    # 训练
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "gradient_clip": 1.0,
    
    # 损失权重
    "render_loss_weight": 0.5,
    "lambda_ssim": 0.8,
    "lambda_l1": 0.2,
    "lambda_scale": 0.001,
    "lambda_opacity": 0.01,
    
    # Flow Matching
    "action_horizon": 16,
    "time_beta_alpha": 1.5,
    "time_beta_beta": 1.0,
    
    # 优化器
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    
    # 数据增强
    "use_augmentation": True,
    "crop_size": 224,
    "color_jitter": 0.2
}
```

---

## 📊 关键参数总结

### 模型参数量

| 模块 | 输入维度 | 输出维度 | 参数量 | 训练状态 |
|------|---------|---------|--------|---------|
| **SigLIP Encoder** | [B,T,224,224,3] | [B,T×256,2048] | ~400M | 冻结 |
| **Gemma Embedding** | [B,L] | [B,L,2048] | ~500M | 冻结 |
| **VGGT Encoder** | [B,T,224,224,3] | [B,T,37,37,2048] | ~1B | 冻结 |
| **GaussianAdapter (LoRA)** | [B,T,37,37,2048] | [B,256,2048] | ~50M | **可训练** |
| **PaliGemma VLM** | [B,N,2048] | [B,N,2048] | ~3B | 冻结 |
| **GaussianDecoder** | [B,256,2048] | [B,256,256,17] | ~100M | **可训练** |
| **GaussianRenderer** | [B,N,17] | [B,3,224,224] | 0 | 无参数 |
| **Action Expert** | [B,N+H,2048] | [B,H,32] | ~1.5B | **可训练** |
| **总计** | - | - | **~6.55B** | **~1.65B可训练** |

### 数据维度

| 数据类型 | 维度 | 说明 |
|---------|------|------|
| 输入图像 | [B, T, 224, 224, 3] | T=3 (历史帧) |
| Vision Tokens | [B, 768, 2048] | 3帧 × 256 tokens |
| Language Tokens | [B, L, 2048] | L ≈ 20-50 |
| Gaussian Tokens | [B, 256, 2048] | Future query tokens |
| Prefix Total | [B, ~1024, 2048] | 所有context tokens |
| Gaussian Params | [B, 256, 256, 17] | 密集参数图 |
| 3D Points | [B, 65536, 3] | 256×256点云 |
| Rendered Image | [B, 3, 224, 224] | RGB图像 |
| Actions | [B, 16, 32] | 16步动作序列 |

### 训练超参数

| 参数 | 值 | 说明 |
|------|---|------|
| Batch Size | 8 | 受GPU内存限制 |
| Learning Rate | 1e-4 | AdamW |
| Gradient Clip | 1.0 | 防止NaN |
| Render Loss Weight | 0.5 | 平衡action和render |
| SSIM Weight | 0.8 | 光度损失 |
| L1 Weight | 0.2 | 光度损失 |
| Scale Regularization | 0.001 | Gaussian尺度 |
| Opacity Regularization | 0.01 | Gaussian不透明度 |
| Action Horizon | 16 | 预测16步 |
| Time Beta α | 1.5 | Flow matching |
| Time Beta β | 1.0 | Flow matching |

---

## ❓ 常见问题 FAQ

### Q1: 为什么用单位矩阵作为viewmatrix?

**A**: depth2pc已经输出相机空间坐标 (z = depth > 0)，相机在原点看向+Z方向，不需要额外的world-to-camera变换。添加平移会把Gaussians推远，导致渲染异常。

### Q2: 为什么scale从0.001改为0.01?

**A**: 
- 0.001太小 → Gaussian覆盖像素少 → 渲染质量差
- 0.01合理 → 符合3DGS标准范围 (0.01-1.0)
- 增大10倍后，Gaussian能更好地覆盖场景

### Q3: 为什么要冻结VLM和VGGT?

**A**:
- VLM和VGGT是预训练模型，已经学到了丰富的视觉-语言知识
- 冻结可以防止灾难性遗忘
- 只训练decoder和adapter，参数量从6.5B降到1.65B，训练更快

### Q4: GaussianDecoder为什么用DPT-style架构?

**A**:
- DPT (Dense Prediction Transformer) 擅长密集预测任务
- 多尺度特征融合能捕捉不同层次的几何信息
- Residual连接稳定训练

### Q5: 为什么用Flow Matching而不是Diffusion?

**A**:
- Flow Matching更简单，不需要复杂的噪声调度
- 推理更快，ODE求解步数少 (10步 vs 50步)
- 训练更稳定，梯度更平滑

### Q6: 如何调节action loss和render loss的权重?

**A**:
```python
# 阶段1: 先训练action (前10k步)
render_loss_weight = 0.0

# 阶段2: 逐渐增加render loss (10k-20k步)
render_loss_weight = 0.1 → 0.5

# 阶段3: 联合训练 (20k步后)
render_loss_weight = 0.5
```

### Q7: 训练时出现NaN怎么办?

**A**:
1. 检查scale参数是否太小 (应该≥0.01)
2. 添加梯度裁剪: `clip_grad_norm_(model.parameters(), 1.0)`
3. 检查depth值范围 (应该在0.1-10.0之间)
4. 降低学习率 (1e-4 → 5e-5)
5. 使用混合精度训练时，确保关键参数用float32

### Q8: 如何可视化训练过程?

**A**:
```python
# 保存渲染对比图
if step % 100 == 0:
    save_image(
        torch.cat([rendered_image, target_image], dim=3),
        f"{vis_dir}/step_{step}.png"
    )

# 打印统计信息
if step % 40 == 0:
    print(f"Scales: min={scales.min():.6f}, max={scales.max():.6f}")
    print(f"Opacity: min={opacity.min():.6f}, max={opacity.max():.6f}")
```

### Q9: 推理时如何加速?

**A**:
1. 使用torch.compile (PyTorch 2.0+)
2. 减少ODE求解步数 (10步 → 5步)
3. 使用半精度 (bfloat16)
4. 批量推理 (batch_size > 1)
5. 缓存VLM特征 (如果prompt不变)

### Q10: 如何适配到新的机器人平台?

**A**:
1. 修改action维度 (32D → 你的机器人DOF)
2. 调整相机参数 (intrinsics, FOV)
3. 收集数据 (至少1000条轨迹)
4. 微调 (冻结VLM, 只训练action expert)
5. 测试泛化性 (新场景, 新物体)

---

## 📚 参考文献

1. **PI0**: Physical Intelligence - Vision-Language-Action Model
2. **3D Gaussian Splatting**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
3. **AD-FFgsStudio**: Autonomous Driving with 4D Gaussian Splatting
4. **PaliGemma**: Google - Vision-Language Model
5. **Flow Matching**: Flow Matching for Generative Modeling
6. **DPT**: Vision Transformers for Dense Prediction

---

## 📝 更新日志

- **2026-03-07**: 
  - 修复scale参数 (0.001 → 0.01)
  - 完善架构文档
  - 添加FAQ和训练流程

- **2026-01-26**:
  - 实现独立GaussianDecoder
  - 添加多视角渲染支持

- **2026-01-24**:
  - 重构世界模型架构
  - 添加DPT-style特征融合

- **2026-01-19**:
  - 初始版本
  - 集成3DGS到PI0

---

**文档结束** 🎉

