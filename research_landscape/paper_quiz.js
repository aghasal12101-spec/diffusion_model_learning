// 自测题与解答。键与 PAPERS 的 k 字段对应。答案写的是"能讲清楚"的程度，不追求最简。
const Q = String.raw;
window.PAPER_QUIZ = {

rf: [
  { q: Q`为什么用独立配对 $(X_0,X_1)$ 训练出的 1-rectified flow，一步生成效果很差？`,
    a: Q`独立配对下，不同样本对的直线插值会互相交叉。在交叉点上 flow matching 学到的是各条直线速度的条件平均，于是真实 ODE 轨迹是弯的。一步 Euler 等于沿起点切线走直线，轨迹越弯误差越大。CIFAR-10 上 1-RF 一步 FID 高达 378。` },
  { q: Q`Reflow 为什么能把轨迹拉直，同时又不改变生成分布？`,
    a: Q`新耦合 $(Z_0,Z_1)$ 由 ODE 生成，是确定性的：每个 $Z_0$ 只对应一个 $Z_1$，直线插值的交叉大大减少，重新训练后的速度场更接近直线。论文证明 rectify 操作保持边缘分布不变，并且对任意凸代价，传输代价不增。` },
  { q: Q`多轮 reflow 的主要代价是什么？后来的工作怎么看？`,
    a: Q`每一轮都要用当前模型解 ODE 生成大量配对，计算昂贵，而且模型误差会随轮数累积。"Improving the Training of Rectified Flows" 指出实际中一轮 reflow 就已接近直线，剩下的误差主要靠更好的时间采样（U 形分布）和距离函数（LPIPS-Huber）来解决。` }
],

cm: [
  { q: Q`一致性函数的参数化为什么要求 $c_{\text{skip}}(\epsilon)=1,\ c_{\text{out}}(\epsilon)=0$？`,
    a: Q`这样边界条件 $f_\theta(x,\epsilon)=x$ 由构造自动成立。如果没有边界条件，网络输出任意常数也能让"相邻两点输出相同"的一致性损失为零。边界条件把一致性的链条锚定在数据端，才使唯一解是真正的 PF-ODE 终点映射。` },
  { q: Q`一致性训练（CT）在没有老师的情况下，怎样得到 PF-ODE 上"相邻"的两个点？`,
    a: Q`用同一个噪声 $z$ 构造 $x_{t_n}=x+t_n z$ 与 $x_{t_{n+1}}=x+t_{n+1}z$。这等价于用单样本 score 估计 $-(x_t-x)/t^2$ 代替老师做一步 Euler。论文证明 $\Delta t\to0$ 时 CT 损失与 CD 损失只差 $o(\Delta t)$。` },
  { q: Q`目标侧为什么要用 $\theta^-$（stop-gradient / EMA），而不是两边都反传？`,
    a: Q`目标和预测来自同一个网络，两边都反传时，最容易的下降方向是让两边一起坍缩成常数。stop-gradient 让目标在当前步固定，变成"自举回归"，训练才稳定。iCT 进一步发现，对 CT 而言目标网络直接取当前参数的 stop-gradient（不用 EMA）效果更好。` }
],

scm: [
  { q: Q`连续时间 CM 为什么要从"梯度的极限"出发，而不是直接取损失的极限？`,
    a: Q`$\Delta t\to0$ 时两个点重合，离散 CM 损失本身趋于 0，没有信息；但除以 $\Delta t$ 后梯度有非零极限，其中包含切线 $\mathrm df_{\theta^-}/\mathrm dt$。实现时把它写成代理损失 $\|F_\theta-\text{sg}(F_{\theta^-}-g)\|^2$，使这个损失的梯度恰好等于极限梯度。切线 $g$ 用一次 JVP 计算。` },
  { q: Q`TrigFlow $x_t=\cos t\,x_0+\sin t\,z$ 带来了什么好处？`,
    a: Q`它把 EDM（方差保持、$\sigma$ 参数化）和 flow matching 统一起来：一致性函数写成 $f=\cos t\,x_t-\sin t\,\sigma_d F_\theta$，系数简单有界，避免了 $\sigma\to\infty$ 时的数值问题，也让切线中各项的大小便于分析。作者正是借此定位不稳定来源，再用恒等时间嵌入、切线归一化和预热修好。` },
  { q: Q`sCM 在 10B 级文生图/视频上细节不如 DMD2，rCM 认为原因是什么？`,
    a: Q`两点：(1) 前向一致性目标让误差从小时间向大时间传播、累积；(2) 前向散度偏向覆盖所有模式，牺牲锐度。DMD 的反向散度是 mode-seeking 的，正好互补，所以 rCM 加了 $\lambda=0.01$ 的 DMD 项作为"长跳"正则。` }
],

meanflow: [
  { q: Q`从平均速度的定义推导 MeanFlow 恒等式。`,
    a: Q`定义 $(t-r)\,u(z_t,r,t)=\int_r^t v(z_\tau,\tau)\,\mathrm d\tau$。固定 $r$，两边对 $t$ 求全导数：左边由乘积法则得 $u+(t-r)\frac{\mathrm d u}{\mathrm dt}$，右边由微积分基本定理得 $v(z_t,t)$。移项得 $u=v-(t-r)\frac{\mathrm du}{\mathrm dt}$。` },
  { q: Q`计算 $\frac{\mathrm d}{\mathrm dt}u(z_t,r,t)$ 的 JVP，切向量为什么是 $(v,0,1)$？`,
    a: Q`由链式法则，$\frac{\mathrm du}{\mathrm dt}=\partial_z u\cdot\frac{\mathrm dz_t}{\mathrm dt}+\partial_r u\cdot\frac{\mathrm dr}{\mathrm dt}+\partial_t u\cdot\frac{\mathrm dt}{\mathrm dt}$。沿轨迹 $\mathrm dz_t/\mathrm dt=v$，$r$ 固定所以 $\mathrm dr/\mathrm dt=0$，最后一项系数为 1。因此对输入 $(z,r,t)$ 的切向量就是 $(v,0,1)$。训练时 $v$ 用条件速度 $\epsilon-x$ 代替。` },
  { q: Q`当 $r=t$ 时目标退化成什么？为什么训练中要保留一部分 $r=t$ 的样本？`,
    a: Q`$(t-r)=0$，目标变成 $v$，损失就是普通的 flow matching。它提供瞬时速度这个"锚"，让 $u$ 在对角线上正确；否则恒等式里的自举项缺少可靠的起点，训练会不稳定。论文的消融显示 $r\neq t$ 的比例要适中（约 25%）。` },
  { q: Q`写出一步采样公式。`,
    a: Q`从 $t=1$ 跳到 $r=0$：$z_0=z_1-(1-0)\,u_\theta(z_1,0,1)$，即 $x=\epsilon-u_\theta(\epsilon,0,1)$。` }
],

imf: [
  { q: Q`原始 MeanFlow 的"目标依赖网络"具体指什么？iMF 怎样把它变成标准回归？`,
    a: Q`原始目标 $u_{\text{tgt}}=v-(t-r)\,\frac{\mathrm du_\theta}{\mathrm dt}$ 里含有网络自身，回归目标随网络变化。iMF 把网络项移到预测一侧：$V_\theta=u_\theta+(t-r)\cdot\text{sg}(\text{JVP})$，去回归固定的 $v=\epsilon-x$。目标与网络无关，成为标准回归（"$v$-loss"）。` },
  { q: Q`为什么要把 CFG 强度做成网络的条件输入？`,
    a: Q`原始 MF 把引导"烤"进训练，$\omega$ 一旦选定，推理时就不能改。论文发现最优 $\omega$ 会随模型大小、训练时长和步数变化，越强的模型偏好越小的 $\omega$。把引导参数当作条件（与 $t,r$ 一样）训练时随机采样，推理时就能自由选择。` }
],

pmf: [
  { q: Q`为什么在 256×256 像素空间里预测 $u$ 会失败，预测 $x$ 却可以？`,
    a: Q`$u$ 是含噪的量，铺满整个高维空间；$x$ 近似落在低维图像流形上。patch 为 16×16×3=768 维时，网络很难拟合满维的噪声量（$u$-预测 FID 164.89），预测流形上的 $x$ 则容易得多（9.56）。在 64×64、patch 只有 48 维时，两种预测都可行（3.80 vs 3.82）。` },
  { q: Q`网络输出 $x_\theta$ 后，怎样得到 MeanFlow 需要的 $u_\theta$？`,
    a: Q`由定义 $x\triangleq z_t-t\cdot u$ 反解得 $u_\theta=(z_t-x_\theta)/t$，再按 iMF 的 $v$-loss 计算 $V_\theta=u_\theta+(t-r)\cdot\text{JVP}_{\text{sg}}$ 并回归 $v$。即"$x$-预测、$v$-损失"。` }
],

tvm: [
  { q: Q`TVM 与 MeanFlow 都学双时间映射，它们在"在哪里匹配速度"上有什么不同？`,
    a: Q`MeanFlow 对起点 $t$ 求全导数，涉及 $\partial_z u\cdot v$ 这一空间 Jacobian 项（Eulerian 型）。TVM 对终点 $s$ 求导，要求 $\frac{\mathrm d}{\mathrm ds}f(x_t,t,s)=u(\psi(x_t,t,s),s)$，即跳到终点时的速度要等于那里的真实速度，只需要时间导数（接近 Lagrangian 型）。` },
  { q: Q`为什么 TVM 要修改 DiT 使其接近 Lipschitz？`,
    a: Q`论文的 $W_2$ 上界依赖模型 Lipschitz 连续；同时把未知的 $u(\psi,s)$ 换成 $u_\theta(x_t+f_\theta,s)$ 这一代理，也要求网络对输入变化不敏感。标准 DiT 训练中激活会爆炸，换成 RMSNorm 作为 QK-Norm、去掉 LayerNorm 的可学习参数等改动后，训练稳定。` }
],

boffi_cm: [
  { q: Q`Eulerian、Lagrangian、Progressive 三类自蒸馏分别利用了 flow map 的哪条性质？`,
    a: Q`Eulerian：$\partial_s X_{s,t}+\nabla X_{s,t}\,b_s=0$，沿轨迹移动起点、终点不变，需要空间导数。Lagrangian：$\partial_t X_{s,t}(x)=b_t(X_{s,t}(x))$，终点沿速度场移动，只需时间导数。Progressive：半群性 $X_{u,t}\circ X_{s,u}=X_{s,t}$，不需要导数。` },
  { q: Q`论文实验中哪一类最好？原因是什么？`,
    a: Q`Lagrangian（LSD）在棋盘格、CIFAR-10、CelebA-64、AFHQ-64 上都最好。它既不含空间导数（梯度方差小），也不依赖从小步自举到大步（误差不累积）。` },
  { q: Q`把 MeanFlow 和 Shortcut 分别归到哪一类？`,
    a: Q`MeanFlow 的恒等式含 $v\,\partial_z u$，属于 Eulerian 型；Shortcut 用"两个小跳等于一个大跳"，属于 Progressive（半群）型。` }
],

imm: [
  { q: Q`为什么说一致性模型是 IMM 的特例？`,
    a: Q`IMM 在一组 $M$ 个粒子上用 MMD 比较两个分布。若每组只有 1 个粒子，且只匹配一阶矩（均值），MMD 就退化成两个输出之间的距离，正是 CM 的逐样本一致性损失。作者据此认为 CM 统计效率低，是它训练不稳定的一个原因。` },
  { q: Q`方法名里的"归纳"体现在哪里？`,
    a: Q`若从较早时间 $r$ 跳到 $s$ 的分布已经正确，就要求从 $t>r$ 跳到 $s$ 的分布与之一致，于是正确性从 $r$ 推广到 $t$。$r$ 随训练逐渐靠近 $t$，就像数学归纳法一步步向外推。` }
],

dmd: [
  { q: Q`为什么 $\nabla_\theta D_{\text{KL}}(p_{\text{fake}}\|p_{\text{real}})$ 可以写成两个 score 之差？`,
    a: Q`$D_{\text{KL}}=\mathbb E_{z}[\log p_{\text{fake}}(G_\theta(z))-\log p_{\text{real}}(G_\theta(z))]$。对 $\theta$ 求导时，$x=G_\theta(z)$ 重参数化，链式法则给出 $(\nabla_x\log p_{\text{fake}}-\nabla_x\log p_{\text{real}})\,\partial x/\partial\theta$；$\log p_{\text{fake}}$ 对 $\theta$ 的显式依赖项期望为零。实际在加噪分布上计算，两个 score 分别由 fake 扩散模型和老师估计。` },
  { q: Q`DMD 为什么还要一个回归损失？`,
    a: Q`反向 KL 是 mode-seeking 的：生成器只要落在老师分布的高密度区域就不受罚，容易坍缩到少数模式。回归到老师 ODE 的（噪声，图像）配对给出全局结构约束，起到模式覆盖的作用。代价是要预先生成大量配对。` },
  { q: Q`fake score 网络为什么必须和生成器同步训练？`,
    a: Q`它要估计的是"当前"生成分布的 score，而生成器每一步都在变。如果 fake 网络跟不上，梯度 $s_{\text{fake}}-s_{\text{real}}$ 就是有偏的，训练会发散。DMD2 的 TTUR（每更新一次生成器就更新多次 fake 网络）正是为此。` }
],

dmd2: [
  { q: Q`DMD2 去掉回归损失后，靠什么保持稳定？`,
    a: Q`双时间尺度更新（TTUR）：每更新一次生成器，就更新约 5 次 fake score 网络，让它准确跟踪生成分布，分布匹配梯度因而可靠。另外加入 GAN 损失，并用 backward simulation 消除多步学生训练与推理的不一致。` },
  { q: Q`为什么加了 GAN 损失后，学生有可能超过老师？`,
    a: Q`DMD 项只让学生去匹配老师的分布，上限就是老师。GAN 判别器直接用真实数据训练，能纠正老师 score 的误差，把学生往真实分布推。ImageNet 64 上 DMD2 一步 FID 1.28，确实好于老师。` }
],

ddmd: [
  { q: Q`写出 Decoupled DMD 的梯度分解，并说明 CA 项为什么能单独驱动"多步变少步"。`,
    a: Q`$-\big[(s^{\text{real}}_{\text{cond}}-s^{\text{fake}}_{\text{cond}})+(\alpha-1)(s^{\text{real}}_{\text{cond}}-s^{\text{real}}_{\text{uncond}})\big]\partial G/\partial\theta$。第二项（CA）是按 CFG 方向外推的信号，直接把样本推向"更像该条件"的方向，产生清晰的结果，而且不依赖 fake 模型。实验中只用 CA 就能完成少步转换。` },
  { q: Q`既然 CA 是主力，DM 项还有什么用？`,
    a: Q`DM 是正则（"盾"）。fake 模型持续跟踪学生分布，当 CA 单独作用产生伪影、过饱和或偏离流形时，real 与 fake 的差会把样本拉回来，保证训练稳定。` }
],

rcm: [
  { q: Q`为什么说 sCM 与 DMD 在散度上是互补的？`,
    a: Q`sCM 的一致性目标对应前向散度，倾向覆盖所有模式，多样性好但细节偏软；DMD 是反向 KL，mode-seeking，画面锐利但容易坍缩、多样性差。两者相加（$\lambda=0.01$）就能在保持多样性的同时修复细节。` }
],

mf_var: [
  { q: Q`条件速度 $v_{\text{cond}}$ 在 MeanFlow 损失中扮演哪两个角色？哪一个导致不稳定？`,
    a: Q`(1) 回归目标：噪声 $v'$ 以加性方式进入残差，与 $\theta$ 无关，是无偏的；(2) JVP 的切向量：噪声经过 $J=(t-r)\partial_{x_t}u_\theta-I$ 乘性放大，产生 $\mathrm{Tr}(J\Sigma_{v'}J^\top)$ 项，使损失不降、梯度方差无界。不稳定来自第二个角色。` },
  { q: Q`按 $\beta$ 扫描切向量混合系数，得到了什么出人意料的结论？`,
    a: Q`梯度 MSE 在 $\beta\approx0.94$ 时最小，但 FID 最好的是 $\beta=0$，即原始的无偏切向量。FID 对偏差的惩罚比梯度 MSE 强得多（约 2.6 倍超线性），所以"方差最优"不等于"生成质量最优"。` }
],

drifting: [
  { q: Q`漂移场为什么必须反对称？只用吸引项 $V^+$ 会怎样？`,
    a: Q`反对称 $V_{p,q}=-V_{q,p}$ 保证 $q=p$ 时 $V=0$，平衡点包含正确解。只用吸引项时，每个样本都被拉向附近真实数据的核加权均值（类似 mean-shift），没有排斥项维持多样性，分布会坍缩。消融中只吸引的 FID 是 177.14，而反对称默认设置是 8.46。` },
  { q: Q`Drifting 的损失值等于什么？为什么？`,
    a: Q`$\mathcal L=\mathbb E\|f_\theta(\epsilon)-\text{sg}(f_\theta(\epsilon)+V)\|^2=\mathbb E\|V\|^2$，因为预测减去目标就是 $-V$。训练让漂移场的平方范数趋于 0，即达到平衡。` },
  { q: Q`为什么要在预训练编码器的特征空间里计算漂移？`,
    a: Q`吸引/排斥依赖核函数里的距离。像素或 VAE 潜空间的欧氏距离与语义相似度不一致，核权重不可靠；预训练特征空间（如 latent-MAE）里的距离更有语义，漂移方向也更有意义。编码器只在训练时使用，推理仍是一次前向。` }
],

wflow: [
  { q: Q`为什么 MMD 和 KL 作为能量泛函都不理想，而 Sinkhorn 散度更好？`,
    a: Q`MMD：两个分布相距较远时核值很小，梯度消失，演化停滞。KL：速度场是 $\nabla\log p-\nabla\log q_t$，需要估计生成分布的 score，有偏且不稳定。Sinkhorn 散度只用样本即可计算，基于熵正则 OT 的重心映射能感知全局分布差异，消融 FID 为 7.29，优于 MMD 的 10.40 和 KL 的 10.17。` },
  { q: Q`W-Flow 训练中生成器怎样始终保持"一步"？`,
    a: Q`每次迭代先算出粒子更新 $x+\eta V(x)$，再训练网络 $f_{\theta^{(k+1)}}$ 回归这个更新后的位置。这相当于把"当前生成器后接一个局部传输"这一复合映射蒸馏回单个网络，所以推理时永远只需一次前向。` }
],

priming: [
  { q: Q`重建预训练为什么选固定噪声水平 $s=0.8$，而不是直接用 $s=1$（纯噪声，和推理输入一样）？`,
    a: Q`$s=1$ 时输入与配对的干净图像相互独立，回归的最优解只是数据均值，学不到有用的映射，性能急剧下降。保留部分图像信息的重建才能学到好的初始化。在 0.6 到 1.0 的扫描中 $s=0.8$ 最好。` },
  { q: Q`作者用"Jacobian 对齐"解释预训练为什么有用，具体指什么？`,
    a: Q`一步梯度下降引起的特征变化为 $\Delta h_\theta\approx-\eta\,J_\theta J_\theta^\top v$，其中 $v$ 是分布损失对特征的梯度。预训练越久，$v$ 越落在 $J_\theta J_\theta^\top$ 响应强的方向上，同样的学习率下特征移动更大，分布后训练也就更高效。` }
],

afm: [
  { q: Q`OT 正则 $\mathbb E\|G(z)-z\|^2$ 为什么能稳定对抗训练？`,
    a: Q`GAN 目标的解不唯一：任何把噪声推前到数据分布的映射都是最优解，生成器会在这些解之间漂移。在对抗目标保证边缘匹配的前提下，再最小化二次传输代价，由 Brenier 定理（高斯源分布、二次代价）可以选出唯一的确定性 OT 映射，优化目标变得确定。` },
  { q: Q`为什么要求噪声 $z$ 与数据 $x$ 同维？`,
    a: Q`二次代价 $\|G(z)-z\|^2$ 要求两者在同一空间，Brenier 定理也针对同一空间的传输。这与 flow 模型的设定一致，论文指出这一约束并不损失一般性。` }
]

};
