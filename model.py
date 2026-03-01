# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        # 修改1: channel 16->32，给各任务分支足够的表达空间
        self.mns = MainNetworkStructure(3, 32)

    def forward(self, x):
        Fout, Ftype = self.mns(x)
        return Fout, Ftype


class MainNetworkStructure(nn.Module):
    def __init__(self, inchannel, channel):
        super(MainNetworkStructure, self).__init__()

        self.conv_in  = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.en1 = RBB(channel)
        self.en2 = RBB(2 * channel)
        self.en3 = RBB(4 * channel)

        self.mid = Mid_BB(8 * channel)

        self.de3 = RBB(4 * channel)
        self.de2 = RBB(2 * channel)
        self.de1 = RBB(channel)

        self.vgg1   = RBB(4 * channel)
        self.vgg2   = RBB(2 * channel)
        self.vgg3   = RBB(1 * channel)

        # 分类头：channel*2 是因为 fc_1 输入来自 channel 维的全局池化
        self.fc_1   = nn.Linear(channel, 64)
        self.fc_2   = nn.Linear(64, 32)
        self.fc_end = nn.Linear(32, 8)

        self.act     = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_e1te2  = nn.Conv2d(1 * channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_e2te3  = nn.Conv2d(2 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_e3tmid = nn.Conv2d(4 * channel, 8 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_midtd3 = nn.Conv2d(8 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_d3td2  = nn.Conv2d(4 * channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_d2td1  = nn.Conv2d(2 * channel, 1 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_vgg_8t4 = nn.Conv2d(8 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_vgg_4t2 = nn.Conv2d(4 * channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_vgg_2t1 = nn.Conv2d(2 * channel, 1 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        # 修改2: 在 en1（全分辨率）增加一个 FEM，去雾关键的透射率/大气光在全局低频里
        # 原来只在 en3（1/4分辨率）做频率增强，高频信息损失太多，效果差
        self.freq_enhance_en1         = FrequencyEnhancementModule(channel)
        self.freq_enhance_bottleneck  = FrequencyEnhancementModule(4 * channel)

        # 修改3&4: NIL 扩展为 4 分支（haze/low/rain/snow），详见 NIL 类注释
        self.mn = NIL(8 * channel)

        # 修改12: 去雾专用解码器旁路
        self.dehaze_decoder = DehazeDecoder(channel)

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, x):

        x_in  = self.conv_in(x)

        # 修改2: en1 之后立即做全分辨率频率增强（对去雾最有效）
        x_en1 = self.en1(x_in)
        x_en1_freq = self.freq_enhance_en1(x_en1)   # 保存频率增强后的 en1，供 DehazeDecoder 使用

        x_en2 = self.en2(self.conv_e1te2(self.maxpool(x_en1_freq)))
        x_en3 = self.en3(self.conv_e2te3(self.maxpool(x_en2)))

        # 原有的 en3 频率增强保留（1/4分辨率，补充中频信息）
        x_fen = self.freq_enhance_bottleneck(x_en3)

        x_mid1, x_mid2 = self.mid(self.conv_e3tmid(self.maxpool(x_fen)))

        # VGG 分类分支（结构不变）
        x_vgg_1    = self.vgg1(self.conv_vgg_8t4(x_mid1))
        x_vgg_2    = self.vgg2(self.conv_vgg_4t2(self.maxpool(x_vgg_1)))
        x_vgg_3    = self.vgg3(self.conv_vgg_2t1(self.maxpool(x_vgg_2)))

        x_vgg_pool    = F.adaptive_avg_pool2d(x_vgg_3, (1, 1))
        x_vgg_4       = x_vgg_pool.view(x_vgg_pool.size(0), -1)

        x_vgg_5       = self.act(self.fc_1(x_vgg_4))
        x_vgg_6       = self.act(self.fc_2(x_vgg_5))
        x_type_logits = self.fc_end(x_vgg_6)

        # Softmax 概率传给 NIL（NIL 需要概率做加权），Logits 返回给 CrossEntropyLoss
        x_type_probs = F.softmax(x_type_logits, dim=1)

        x_mid2 = self.mn(x_mid2, x_type_probs)

        x_de3 = self.de3(self.conv_midtd3(self._upsample(x_mid2, x_en3)) + x_en3)
        x_de2 = self.de2(self.conv_d3td2(self._upsample(x_de3, x_en2)) + x_en2)
        x_de1 = self.de1(self.conv_d2td1(self._upsample(x_de2, x_en1_freq)) + x_en1_freq)

        # 修改12: 去雾旁路，按 haze 置信度动态加权叠加到主解码器输出上
        # haze_weight = 含雾类别的概率之和（type 1,5,6,7 都含雾）
        # shape: (B,) -> (B,1,1,1) 广播
        haze_weight = (x_type_probs[:, 1] + x_type_probs[:, 5] +
                       x_type_probs[:, 6] + x_type_probs[:, 7]).view(-1, 1, 1, 1)
        dehaze_residual = self.dehaze_decoder(x_en1_freq, x_en2)   # (B,channel,H,W)
        x_de1 = x_de1 + haze_weight * dehaze_residual              # 只在含雾时生效

        x_out = self.conv_out(x_de1)

        return x_out, x_type_logits


class RBB(nn.Module):
    def __init__(self, channel):
        super(RBB, self).__init__()

        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.act  = nn.PReLU(channel)
        # 修改8: GroupNorm num_groups 从 1 改为合理值，1 等同于 LayerNorm，
        # 对图像任务效果偏弱；改为 min(channel//4, 8) 更接近 GroupNorm 的正常用法
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=min(channel // 4, 8))

    def forward(self, x):
        # 修改8: 顺序改为 Conv -> Norm -> Act（BN/GN 标准顺序）
        # 原代码 act(norm(conv(x))) 顺序实际是对的，但 norm 的 groups=1 相当于无效归一化
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_3 = self.act(self.norm(self.conv_3(x_2)) + x)

        return x_3


class HazeBranch(nn.Module):
    """
    修改9: 专为去雾设计的分支，替代原来的 RBB
    -------------------------------------------------
    问题根源：去雾需要估计全图的大气光 A 和透射率 T（都是全局/低频信息），
    而 RBB 只有 3x3 局部卷积，感受野极小，根本无法感知全图雾的浓度分布。
    这就是为什么纯雾、雾+雨、雾+雪的效果都差——haze 分支能力不足。

    解决方案：局部特征（RBB）+ 全局上下文门控（Global Context Gate）
    - 局部分支：保留 RBB 做局部纹理细节
    - 全局分支：AdaptiveAvgPool 压缩到 1x1 -> 全连接估计全图雾参数
               -> Sigmoid 门控调制局部特征（类似 SENet 的 channel attention）
    - 两路相加后再融合，让网络既看局部细节又感知全局雾浓度
    """
    def __init__(self, channels):
        super(HazeBranch, self).__init__()

        # 局部分支：保留原来 RBB 的局部特征提取能力
        self.local_branch = RBB(channels)

        # 全局上下文门控：压缩空间维度到 1x1，估计全图大气光强度
        # 瓶颈结构降低参数量：channels -> channels//4 -> channels
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B,C,H,W) -> (B,C,1,1)
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()                                       # 输出 0~1 的调制权重
        )

        # 融合层：把局部特征和全局门控后的特征合并
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(num_channels=channels, num_groups=min(channels // 4, 8)),
            nn.PReLU(channels)
        )

    def forward(self, x):
        local_f  = self.local_branch(x)          # (B,C,H,W) 局部特征
        global_w = self.global_context(x)        # (B,C,1,1) 全局权重，自动广播
        # 全局门控调制局部特征，再加残差
        out = self.fuse(local_f * global_w + local_f)
        return out


class DehazeDecoder(nn.Module):
    """
    修改12: 去雾专用解码器旁路
    -------------------------------------------------
    问题：主解码器（de3/de2/de1）是所有退化类型共用的，
    去雾需要的全局亮度/对比度校正能力被其他任务的梯度稀释。

    解决方案：在主解码器旁边增加一条轻量级去雾旁路，
    直接从 en1/en2 的频率增强特征（低频信息最丰富的地方）做补充复原。
    最终输出 = 主解码器输出 + haze_weight * 旁路输出（按分类置信度动态加权）。

    旁路只在含雾类别（haze_weight > 0）时生效，不影响其他任务。
    """
    def __init__(self, channel):
        super(DehazeDecoder, self).__init__()

        # 从 en2 特征（1/2分辨率）提取雾相关信息
        self.dehaze_en2 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 3, 1, 1, bias=False),
            nn.GroupNorm(num_channels=channel, num_groups=min(channel // 4, 8)),
            nn.PReLU(channel)
        )

        # 从 en1 特征（全分辨率）做最终雾校正
        self.dehaze_en1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.GroupNorm(num_channels=channel, num_groups=min(channel // 4, 8)),
            nn.PReLU(channel)
        )

        # 输出层：生成和主解码器相同通道数的残差
        self.dehaze_out = nn.Conv2d(channel, channel, 1, bias=False)

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, x_en1_freq, x_en2):
        """
        x_en1_freq: en1 经过全分辨率 FEM 后的特征（低频信息最丰富）
        x_en2:      en2 的编码特征
        """
        # en2 -> 降通道 -> 上采样到 en1 分辨率
        d2 = self.dehaze_en2(x_en2)
        d2_up = self._upsample(d2, x_en1_freq)

        # 融合 en1 频率特征和上采样的 en2 特征
        d1 = self.dehaze_en1(d2_up + x_en1_freq)

        out = self.dehaze_out(d1)
        return out


class Mid_BB(nn.Module):
    def __init__(self, channel):
        super(Mid_BB, self).__init__()

        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        # 修改7: 为两个输出分支各自增加独立的前置 1x1 卷积
        # 原来两个分支共享 conv_1/conv_2，分类和复原任务梯度方向可能冲突
        self.conv_branch_cls  = nn.Conv2d(channel, channel, kernel_size=1, bias=False)  # 分类分支独立投影
        self.conv_branch_dec  = nn.Conv2d(channel, channel, kernel_size=1, bias=False)  # 解码分支独立投影

        # 分类分支（结构不变）
        self.conv_3_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        # 修改6: 解码分支改为多尺度空洞卷积并联（dilation=1,3,5）
        # 原来只有 dilation=3，感受野 7x7，对雨线/雪花的长程结构捕捉不足
        # 多尺度并联后感受野覆盖 3x3 / 7x7 / 11x11，对不同粗细的雨线和雪花都有效
        self.conv_3_2_d1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,  dilation=1, bias=False)
        self.conv_3_2_d3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3,  dilation=3, bias=False)
        self.conv_3_2_d5 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=5,  dilation=5, bias=False)
        # 融合三个尺度
        self.conv_3_2_fuse = nn.Conv2d(channel * 3, channel, kernel_size=1, bias=False)

        self.act  = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=min(channel // 4, 8))

    def forward(self, x):

        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))

        # 修改7: 两个分支各自走独立投影，解耦梯度
        x_cls = self.conv_branch_cls(x_2)
        x_dec = self.conv_branch_dec(x_2)

        # 分类分支（给 VGG head 用）
        x_3_1 = self.act(self.norm(self.conv_3_1(x_cls)))

        # 修改6: 解码分支多尺度空洞卷积并联
        d1 = self.conv_3_2_d1(x_dec)
        d3 = self.conv_3_2_d3(x_dec)
        d5 = self.conv_3_2_d5(x_dec)
        x_3_2 = self.act(self.norm(self.conv_3_2_fuse(torch.cat([d1, d3, d5], dim=1))))

        return x_3_1, x_3_2


class FrequencyEnhancementModule(nn.Module):
    def __init__(self, channels):
        super(FrequencyEnhancementModule, self).__init__()

        self.low_pass = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)
        self.init_gaussian_kernel()

        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(channels)
        )

        self.low_freq_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(channels)
        )

        self.fusion_adjust = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.PReLU(channels)
        )

        self.norm = nn.GroupNorm(num_channels=channels, num_groups=min(channels // 4, 8))

    def init_gaussian_kernel(self):
        kernel = torch.tensor([
            [1,  4,  6,  4,  1],
            [4, 16, 24, 16,  4],
            [6, 24, 36, 24,  6],
            [4, 16, 24, 16,  4],
            [1,  4,  6,  4,  1]
        ], dtype=torch.float32) / 256.0

        kernel = kernel.view(1, 1, 5, 5)
        kernel = kernel.repeat(self.low_pass.out_channels, 1, 1, 1)
        self.low_pass.weight.data = kernel
        self.low_pass.weight.requires_grad = False

    def forward(self, x):
        low_freq  = self.low_pass(x)
        high_freq = x - low_freq

        low_enhanced  = self.low_freq_adjust(low_freq)
        high_enhanced = self.high_freq_enhance(high_freq)

        out = self.fusion_adjust(torch.cat((low_enhanced, high_enhanced), 1)) + x

        return out


class NIL(nn.Module):
    """
    修改3: 先验矩阵替代盲目 type_mapper
    -----------------------------------------
    原来 type_mapper(Linear 8->16->3 + Softmax) 让网络自己学 8类->3权重的映射，
    但语义完全模糊（snow 应该映射到 haze/low/rain 哪个分支？网络不知道）。
    改为用物理先验矩阵直接定义 8类 -> 4分支(haze/low/rain/snow) 的对应关系，
    再让网络在此基础上学习残差修正，语义清晰，收敛更快。

    修改4: 新增 snow 专用分支 ms
    -----------------------------------------
    原来 NIL 只有 haze/low/rain 三个分支，雪（snow、rainsnow、hazesnow...）
    只能混用 rain 分支，去雪效果差。新增独立 snow 分支，并相应扩展 CrossAttentionModule。

    8类先验映射 (每行和为1):
      0=clear:         [0.25, 0.25, 0.25, 0.25]  → 均匀
      1=haze:          [1.0,  0.0,  0.0,  0.0 ]  → 纯雾
      2=rain:          [0.0,  0.0,  1.0,  0.0 ]  → 纯雨
      3=snow:          [0.0,  0.0,  0.0,  1.0 ]  → 纯雪
      4=rainsnow:      [0.0,  0.0,  0.5,  0.5 ]  → 雨雪混合
      5=hazerain:      [0.5,  0.0,  0.5,  0.0 ]  → 雾雨混合
      6=hazesnow:      [0.5,  0.0,  0.0,  0.5 ]  → 雾雪混合
      7=hazerainsnow:  [0.34, 0.0,  0.33, 0.33]  → 三者混合
    """
    def __init__(self, channels):
        super(NIL, self).__init__()

        # 修改10: mh 换成 HazeBranch（带全局上下文的去雾专用分支）
        # 原来 RBB 感受野太小，无法估计全局大气光，导致所有含雾类别效果都差
        self.mh = HazeBranch(channels)  # haze 专用分支（全局感知）
        self.ml = RBB(channels)         # low-light 专用分支
        self.mr = RBB(channels)         # rain 专用分支
        self.ms = RBB(channels)         # snow 专用分支

        # 修改4: CrossAttentionModule 扩展为 4 分支
        self.cross_attention = CrossAttentionModule(channels, num_branches=4)

        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(channels)
        )

        # 修改11: 先验矩阵更新 —— 含雾类别的 haze 权重提升
        # 原来 hazerain=[0.5,0,0.5,0], hazesnow=[0.5,0,0,0.5]，haze 只占 0.5
        # 但雾是全局效果，在混合退化中处理优先级应更高
        # 新策略：haze 权重提升到 0.65，其余退化均分剩余 0.35
        prior = torch.tensor([
            [0.25, 0.25, 0.25, 0.25],   # 0: clear       → 均匀
            [1.0,  0.0,  0.0,  0.0 ],   # 1: haze        → 纯雾，全给 haze 分支
            [0.0,  0.0,  1.0,  0.0 ],   # 2: rain        → 纯雨
            [0.0,  0.0,  0.0,  1.0 ],   # 3: snow        → 纯雪
            [0.0,  0.0,  0.5,  0.5 ],   # 4: rainsnow    → 不含雾，雨雪平分
            [0.65, 0.0,  0.35, 0.0 ],   # 5: hazerain    → 雾优先(0.65)，雨补充(0.35)
            [0.65, 0.0,  0.0,  0.35],   # 6: hazesnow    → 雾优先(0.65)，雪补充(0.35)
            [0.55, 0.0,  0.23, 0.22],   # 7: hazerainsnow→ 雾最高(0.55)，雨雪各约0.22
        ], dtype=torch.float32)
        self.register_buffer('prior_map', prior)

        # 残差修正网络（在先验基础上微调）
        self.residual_mapper = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x, x_type):
        # x_type: (B, 8) 概率

        # 先验权重: (B, 4) = (B, 8) @ (8, 4)
        prior_weights = x_type @ self.prior_map

        # 残差修正: (B, 4)
        residual = self.residual_mapper(x_type)

        # 最终权重（先验 + 残差，再归一化确保和为1）
        degradation_weights = F.softmax(prior_weights + residual, dim=1)  # (B, 4)

        B, C, H, W = x.shape

        haze_p = self.mh(x)
        low_p  = self.ml(x)
        rain_p = self.mr(x)
        snow_p = self.ms(x)   # 修改4: snow 分支

        # 修改4: 4分支交叉注意力
        enhanced = self.cross_attention(haze_p, low_p, rain_p, snow_p)

        weights   = degradation_weights.view(B, 4, 1, 1)
        weighted_o = (
            haze_p * weights[:, 0:1] +
            low_p  * weights[:, 1:2] +
            rain_p * weights[:, 2:3] +
            snow_p * weights[:, 3:4]   # 修改4: snow 加权
        )

        output = self.adaptive_fusion(enhanced + weighted_o)

        return output


class CrossAttentionModule(nn.Module):
    """
    修改4: 扩展为支持 num_branches 个分支（默认4，对应 haze/low/rain/snow）
    原来硬编码为 3 分支，现在改为参数化，方便后续调整。
    """
    def __init__(self, channels, reduction_ratio=4, num_branches=4):
        super(CrossAttentionModule, self).__init__()

        self.num_branches   = num_branches
        reduced_channels    = max(channels // reduction_ratio, 1)
        self.reduced_channels = reduced_channels

        self.channel_reduce = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.channel_expand = nn.Conv2d(reduced_channels * num_branches, channels, kernel_size=1)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(reduced_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # groups=num_branches 保证各分支在 cross_fusion 里相对独立处理
        self.cross_fusion = nn.Conv2d(
            reduced_channels * num_branches,
            reduced_channels * num_branches,
            kernel_size=3, padding=1, groups=num_branches
        )
        self.unified_modulation = nn.Sequential(
            nn.Conv2d(reduced_channels * num_branches, reduced_channels * num_branches, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, *branch_features):
        """
        branch_features: num_branches 个 tensor，每个 shape=(B,C,H,W)
        """
        assert len(branch_features) == self.num_branches

        # 降维
        reduced = [self.channel_reduce(f) for f in branch_features]  # list of (B, rc, H, W)

        # 交叉增强: 每个分支 = 自身 + 0.5 * 其余所有分支均值
        enhanced_list = []
        for i, r in enumerate(reduced):
            others = [reduced[j] for j in range(self.num_branches) if j != i]
            others_mean = sum(others) / len(others)
            enhanced_list.append(r + 0.5 * others_mean)

        # 空间注意力
        spatial_list = [self.spatial_conv(e) for e in enhanced_list]

        # 统一门控
        all_f         = torch.cat(reduced, dim=1)          # (B, rc*N, H, W)
        fused_all     = self.cross_fusion(all_f)
        unified_gate  = self.unified_modulation(fused_all) # (B, rc*N, H, W)

        rc = self.reduced_channels
        out_list = []
        for i, r in enumerate(reduced):
            gate = unified_gate[:, i*rc:(i+1)*rc, :, :]
            out_list.append(r * spatial_list[i] * gate + r)

        out = self.channel_expand(torch.cat(out_list, dim=1))

        return out
