// 论文解读数据。每条以 k 对应 few_step_generation.html 中 PAPERS 的 k 字段。
// src: "full" = 依据原文全文整理；"abstract" = 只读到摘要/部分内容；"known" = 经典工作，依据整理者对原文的熟悉程度撰写，细节请对照原文。
const R = String.raw;
window.PAPER_DETAILS = window.PAPER_DETAILS || {};
Object.assign(window.PAPER_DETAILS, {

cm: {
  src: "known",
  tldr: R`让网络直接学习 PF-ODE 上任意一点到轨迹起点的映射 $f_\theta(x_t,t)\approx x_0$，用"同一条轨迹上相邻两点的输出应相同"作为训练信号，实现一步生成。`,
  motivation: R`扩散模型采样需要几十到上千次网络调用。之前的加速办法要么是更好的 ODE 求解器（10 步左右就明显掉质量），要么是渐进蒸馏（需要多轮训练）。作者希望有一个模型族，天然支持一步生成，又能用多步采样换质量。`,
  method: R`
<p><b>参数化。</b>用 EDM 风格的跳连写成
$$f_\theta(x,t)=c_{\text{skip}}(t)\,x+c_{\text{out}}(t)\,F_\theta(x,t),$$
其中 $c_{\text{skip}}(\epsilon)=1,\ c_{\text{out}}(\epsilon)=0$，于是边界条件 $f_\theta(x,\epsilon)=x$ 自动成立。</p>
<p><b>一致性蒸馏（CD）。</b>把 $[\epsilon,T]$ 离散成 $t_1\lt\dots\lt t_N$。取数据 $x$，加噪得 $x_{t_{n+1}}$，用老师的 ODE 求解器走一步得到 $\hat x_{t_n}$，然后最小化
$$\mathcal L_{\text{CD}}=\mathbb E\big[\lambda(t_n)\,d\big(f_\theta(x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat x_{t_n},t_n)\big)\big],$$
$\theta^-$ 是 $\theta$ 的 EMA（目标网络），$d$ 可以是 $\ell_2$ 或 LPIPS。</p>
<p><b>一致性训练（CT）。</b>不用老师：利用 $x_t=x+t\,z$ 时 score 的无偏估计 $-(x_t-x)/t^2$，等价于直接用同一个噪声 $z$ 构造 $x_{t_n}=x+t_n z$ 和 $x_{t_{n+1}}=x+t_{n+1}z$ 两点。需要随训练逐步增加离散步数 $N$ 的课程。</p>
<p><b>多步采样。</b>一步得到 $\hat x_0$ 后，重新加噪到较小的 $\tau$，再用 $f_\theta$ 去噪，交替进行，可以用算力换质量。</p>`,
  algo: [
    "for each step:",
    "  x ~ data, n ~ U{1..N-1}, z ~ N(0,I)",
    "  x_next = x + t[n+1]·z",
    "  CD: x_prev = ODESolver_teacher(x_next, t[n+1] → t[n])",
    "  CT: x_prev = x + t[n]·z",
    "  loss = λ(t[n]) · d(f_θ(x_next, t[n+1]), f_θ⁻(x_prev, t[n]))",
    "  θ ← θ − η∇loss;  θ⁻ ← μθ⁻ + (1−μ)θ"
  ],
  theory: R`若 CD 损失为 0 且 ODE 求解器的局部误差为 $O(\Delta t^{p+1})$，则 $\sup\|f_\theta(x,t)-f(x,t)\|=O(\Delta t^p)$，即一致性函数被准确学到，误差只来自离散化。CT 损失在 $\Delta t\to0$ 时与 CD 损失等价（差一个 $o(\Delta t)$ 项）。`,
  results: { head: ["设置", "数据集", "NFE", "FID"], rows: [
    ["CD (LPIPS)", "CIFAR-10", "1", "3.55"], ["CD (LPIPS)", "CIFAR-10", "2", "2.93"],
    ["CT", "CIFAR-10", "1", "8.70"], ["CD (LPIPS)", "ImageNet 64", "1", "6.20"], ["CD (LPIPS)", "ImageNet 64", "2", "4.70"]
  ], note: "CD 明显好于 CT；LPIPS 作为距离比 ℓ2 好很多，但也被认为会让 FID 偏乐观。" },
  limits: R`CT 训练不稳定、对离散化课程和 EMA 很敏感；依赖 LPIPS；一步质量与多步扩散仍有差距。这些问题直接引出了 iCT、ECT、sCM。`,
  relations: R`后续 sCM 把它连续时间化；IMM 证明它是"单粒子一阶矩匹配"的特例；在 flow map 框架下，CM 只学了 $X_{t\to0}$ 这一列。`
},

ect: {
  src: "known",
  tldr: R`从预训练扩散模型出发做一致性训练，并把离散步长 $\Delta t$ 从"整段"逐渐缩小到接近 0，大幅降低 CM 的训练成本。`,
  motivation: R`从头做 CT 需要大量算力，而且早期训练很不稳定。作者观察到：$\Delta t$ 取最大时一致性条件退化为普通去噪（扩散训练），$\Delta t\to0$ 时才是真正的一致性条件。所以扩散训练本身可以看作 CT 的一个特例。`,
  method: R`
<p>把一致性条件写成关于 $\Delta t$ 的连续族：
$$\mathcal L(\theta;\Delta t)=\mathbb E\big[w(t)\,d\big(f_\theta(x_t,t),\,f_{\text{sg}(\theta)}(x_{t-\Delta t},t-\Delta t)\big)\big].$$
当 $t-\Delta t=0$ 时目标是干净数据 $x$，就是去噪损失。</p>
<p>训练时先加载预训练扩散模型权重（对应 $\Delta t=t$ 的情形），再按课程把 $\Delta t$ 逐步缩小（"tighten"），同时设计了与 $\Delta t$ 相关的权重和更换为伪 Huber 距离等细节。</p>`,
  algo: [
    "θ ← 预训练扩散模型权重",
    "for stage k = 1..K:  Δt 的上限按课程缩小",
    "  x ~ data, t ~ p(t), z ~ N(0,I)",
    "  x_t = x + t·z,  x_r = x + (t−Δt)·z",
    "  loss = w(t,Δt) · d(f_θ(x_t,t), sg(f_θ(x_r,t−Δt)))",
    "  更新 θ"
  ],
  theory: R`论文主要贡献是视角上的：把扩散训练与一致性训练放在同一个以 $\Delta t$ 为参数的连续族里，并从微分形式的一致性条件解释为什么逐步收紧是合理的。`,
  results: { head: ["数据集", "NFE", "说明"], rows: [
    ["CIFAR-10", "2", "FID 2.73，单张 A100 约 1 小时，与数百 GPU 小时的一致性蒸馏相当"], ["规模化", "—", "ECT 下 CM 的表现符合经典的幂律缩放"]
  ], note: "2024 年时在 CIFAR-10 上训练最好的 CM 需要 8 卡一周。" },
  limits: R`依赖预训练扩散模型作为初始化，因此并非完全"从头"；仍是离散时间 CM，后续被 sCM 的连续时间形式部分取代。`,
  relations: R`和 sCM 一起奠定了"连续时间一致性"的思路；与 MeanFlow 的"从 $r=t$（即 flow matching）逐步过渡到 $r\neq t$"有相似的精神。`
},

scm: {
  src: "known",
  tldr: R`把一致性模型彻底改成连续时间形式（TrigFlow 参数化 + JVP 计算切线），系统解决训练不稳定问题，首次把 CM 扩展到 1.5B 参数和 ImageNet 512。`,
  motivation: R`离散时间 CM 有离散化误差，需要调步数课程；以往的连续时间 CM 尝试则会训练发散。作者要找出发散的根源并修好，使连续时间 CM 可扩展。`,
  method: R`
<p><b>TrigFlow。</b>统一 EDM 和 flow matching：$x_t=\cos t\,x_0+\sin t\,z$，$t\in[0,\pi/2]$，$z\sim\mathcal N(0,\sigma_d^2 I)$。一致性函数取
$$f_\theta(x_t,t)=\cos t\,x_t-\sin t\,\sigma_d\,F_\theta(x_t/\sigma_d,\,t).$$</p>
<p><b>连续时间目标。</b>离散 CM 损失在 $\Delta t\to0$ 时的梯度为
$$\nabla_\theta\,\mathbb E\Big[w(t)\,F_\theta^\top\,\frac{\mathrm d f_{\theta^-}(x_t,t)}{\mathrm d t}\Big],$$
其中切线 $\frac{\mathrm d f}{\mathrm dt}$ 需要沿 PF-ODE 求全导数，用一次前向模式自动微分（JVP）算出。sCD 用老师给出 $\mathrm dx_t/\mathrm dt$，sCT 用 $\cos t\,z-\sin t\,x_0$ 的无偏估计。</p>
<p><b>稳定化技巧。</b>(1) 时间嵌入用恒等变换 $c_{\text{noise}}(t)=t$，避免 $\log\tan$ 之类在边界爆炸；(2) 位置嵌入替代傅里叶嵌入；(3) 自适应双重归一化；(4) 切线归一化 / 裁剪；(5) 切线预热（逐渐打开切线里不稳定的项）；(6) 学习的自适应损失权重。并实现了支持 JVP 的 FlashAttention。</p>`,
  algo: [
    "x0 ~ data, z ~ N(0,σ_d²I), t ~ p(t)",
    "x_t = cos t·x0 + sin t·z",
    "dx_t/dt = 老师速度 (sCD) 或 cos t·z − sin t·x0 (sCT)",
    "(F, dF/dt) = JVP(F_θ⁻, (x_t/σ_d, t), (dx_t/dt/σ_d, 1))",
    "g = 由 F 与 dF/dt 组合得到的切线，归一化 g/(‖g‖+c)",
    "loss = ‖F_θ − sg(F_θ⁻ − g)‖² · 自适应权重",
    "更新 θ"
  ],
  theory: R`推导了离散 CM 梯度在 $\Delta t\to0$ 的极限，指出切线中 $\sin t\,\partial_t F$ 等项在 $t\to\pi/2$ 附近会造成不稳定，从而对应地设计了时间嵌入和预热策略。`,
  results: { head: ["数据集", "模型", "NFE", "FID"], rows: [
    ["CIFAR-10", "sCT", "2", "2.06"], ["ImageNet 64", "sCD", "2", "1.48"], ["ImageNet 512", "sCD-XXL 1.5B", "2", "1.88"]
  ], note: "两步结果与最好的扩散模型差距在 10% 以内。" },
  limits: R`前向散度目标偏向覆盖模式，细节（如文字、精细纹理）不如 DMD 一类；JVP 在大模型上工程代价高。rCM 针对的正是这两点。`,
  relations: R`MeanFlow 的恒等式与 sCM 的切线本质上都是对 flow map 的时间全导数；rCM 在 sCM 上加了 score 蒸馏正则。`
},

fmm: {
  src: "known",
  tldr: R`把少步生成统一为学习双时间流映射 $X_{s,t}$：把任意时刻 $s$ 的点沿概率流送到时刻 $t$，给出直接从数据学 $X_{s,t}$ 的损失，以及从已有速度场蒸馏的两类损失。`,
  motivation: R`一致性模型只学到"到 0 时刻"的映射，渐进蒸馏只学固定步长。作者希望有一个统一对象，同时覆盖一步、少步和任意步长采样，并且有清晰的数学刻画。`,
  method: R`
<p>对随机插值 $I_t=\alpha_t x_0+\beta_t x_1$ 及其速度场 $b_t$，流映射满足 $X_{s,t}(x_s)=x_t$，并有三个刻画：</p>
<ul>
<li>Lagrangian 方程：$\partial_t X_{s,t}(x)=b_t\big(X_{s,t}(x)\big)$；</li>
<li>Eulerian 方程：$\partial_s X_{s,t}(x)+b_s(x)\cdot\nabla X_{s,t}(x)=0$；</li>
<li>半群性：$X_{u,t}\circ X_{s,u}=X_{s,t}$，且 $X_{s,s}=\text{id}$。</li>
</ul>
<p>参数化 $X_{s,t}(x)=x+(t-s)\,v_{s,t}(x)$，自动满足 $X_{s,s}=\text{id}$，且 $v_{t,t}=b_t$。据此提出：(1) <b>Flow Map Matching</b>，直接用插值数据训练；(2) <b>Lagrangian / Eulerian Map Distillation</b>，以预训练的 $b_t$ 为老师，最小化上述方程的残差。</p>`,
  algo: [
    "s, t ~ p(s,t); x0, x1 ~ ρ0, ρ1; x_s = I_s(x0,x1)",
    "LMD: loss = ‖∂_t X_θ(s,t,x_s) − b_t(X_θ(s,t,x_s))‖²",
    "EMD: loss = ‖∂_s X_θ + (∇X_θ)·b_s(x_s)‖²",
    "∂_t / ∇ 项用 JVP 计算",
    "采样: x ← X_θ(t_k, t_{k+1}, x) 按任意时间网格迭代"
  ],
  theory: R`证明了上述损失的极小值点恰为真实流映射，并给出用流映射误差控制生成分布 Wasserstein 距离的界。`,
  results: { head: ["实验", "说明"], rows: [
    ["CIFAR-10 / ImageNet 32", "少步采样明显优于同等步数的 ODE 求解；与 CM、渐进蒸馏对比"]
  ], note: "这篇以框架和理论为主，具体数字请对照原文。" },
  limits: R`实验规模较小；直接的 Flow Map Matching 损失需要知道配对样本，实践中常用蒸馏版本。`,
  relations: R`这是后来"自蒸馏"、MeanFlow、AYF、TVM 共同的理论基础。MeanFlow 的平均速度 $u(z_t,r,t)$ 正是这里的 $v_{s,t}$（时间方向相反）。`
},

shortcut: {
  src: "known",
  tldr: R`把步长 $d$ 也作为网络输入，学习 $s_\theta(x_t,t,d)$。用"两个步长为 $d$ 的小跳等于一个步长为 $2d$ 的大跳"做自洽训练，单阶段、单网络支持任意步数采样。`,
  motivation: R`CM、渐进蒸馏都需要多阶段训练、老师模型或复杂的步长课程。作者想要一个只训练一次、无需老师、推理时随意选步数的方法。`,
  method: R`
<p>模型预测从 $x_t$ 出发步长为 $d$ 的"捷径方向"，一步更新为 $x_{t+d}=x_t+d\,s_\theta(x_t,t,d)$。</p>
<p><b>训练目标（两部分混合）。</b></p>
<ul>
<li>$d=0$：普通 flow matching，$\|s_\theta(x_t,t,0)-(x_1-x_0)\|^2$；</li>
<li>$d>0$：自洽目标
$$s_{\text{target}}=\tfrac12\big[s_\theta(x_t,t,d)+s_\theta(x'_{t+d},t+d,d)\big],\quad x'_{t+d}=x_t+d\,s_\theta(x_t,t,d),$$
最小化 $\|s_\theta(x_t,t,2d)-\text{sg}(s_{\text{target}})\|^2$。</li>
</ul>
<p>步长取 2 的幂次。一个批次里约 1/4 用于自洽目标、3/4 用于 flow matching。</p>`,
  algo: [
    "x0 ~ N, x1 ~ data, t ~ U, x_t = (1−t)x0 + t·x1",
    "batch 的 3/4: loss_FM = ‖s_θ(x_t,t,0) − (x1−x0)‖²",
    "batch 的 1/4: d ~ {1/128,...,1/2}",
    "  s1 = s_θ(x_t,t,d); x' = x_t + d·s1; s2 = s_θ(x',t+d,d)",
    "  loss_SC = ‖s_θ(x_t,t,2d) − sg((s1+s2)/2)‖²",
    "更新 θ"
  ],
  theory: R`$d\to0$ 时退化为 flow matching，保证了速度场的正确性；$d>0$ 的目标是流映射半群性的离散化。论文本身以经验结果为主。`,
  results: { head: ["数据集", "说明"], rows: [
    ["CelebA-HQ 256", "1 步与多步都优于同样单阶段训练的 CM、ReFlow 基线"],
    ["ImageNet 256", "1 步质量可用，但与多步扩散仍有明显差距"]
  ], note: "具体数字请对照原文表格。" },
  limits: R`一步质量距离后来的 MeanFlow 等方法差距较大；离散的步长集合与自洽误差的累积是主要瓶颈。`,
  relations: R`在 Boffi 等的自蒸馏框架里属于 Progressive（半群）类；MeanFlow 可以看作它的连续步长、微分形式版本。`
},

meanflow: {
  src: "known",
  tldr: R`定义区间 $[r,t]$ 上的平均速度 $u$，推导出 $u$ 与瞬时速度 $v$ 之间的恒等式，把它变成可直接训练的回归目标，从头训练出一步生成模型（ImageNet 256 上 1-NFE FID 3.43）。`,
  motivation: R`Flow matching 学的是瞬时速度，一步积分误差大。CM、Shortcut 等都靠人为设计的一致性约束，训练不够稳。作者希望找一个有明确定义、能直接回归的"一步量"。`,
  method: R`
<p><b>平均速度。</b>
$$u(z_t,r,t)\triangleq\frac1{t-r}\int_r^t v(z_\tau,\tau)\,\mathrm d\tau,\qquad z_r=z_t-(t-r)\,u(z_t,r,t).$$</p>
<p><b>MeanFlow 恒等式。</b>对 $(t-r)u=\int_r^t v$ 两边关于 $t$ 求全导数：
$$u(z_t,r,t)=v(z_t,t)-(t-r)\,\frac{\mathrm d}{\mathrm dt}u(z_t,r,t),\quad \frac{\mathrm d u}{\mathrm dt}=v\,\partial_z u+\partial_t u.$$
全导数用一次 JVP 计算，切向量为 $(v,0,1)$，分别对应 $(z,r,t)$。</p>
<p><b>损失。</b>
$$\mathcal L=\mathbb E\,\big\|u_\theta(z_t,r,t)-\text{sg}(u_{\text{tgt}})\big\|^2,\quad u_{\text{tgt}}=v_t-(t-r)\big(v_t\,\partial_z u_\theta+\partial_t u_\theta\big),$$
其中 $v_t=\epsilon-x$ 是条件速度。使用自适应权重 $w=1/(\|\Delta\|^2+c)^p$。$r=t$ 时退化为 flow matching，训练中一部分样本令 $r=t$。</p>
<p><b>无 CFG 推理的引导。</b>训练时把目标速度替换为 $\tilde v_t=\omega\,v_t+(1-\omega)\,u_\theta(z_t,t,t)$（无条件版本），引导被"烤"进模型，采样仍是 1 NFE。</p>
<p><b>采样。</b>$x=\epsilon-u_\theta(\epsilon,0,1)$。</p>`,
  algo: [
    "x ~ data, ε ~ N(0,I); (t, r) ~ logit-normal, 令一部分 r = t",
    "z = (1−t)x + tε,  v = ε − x",
    "(u, dudt) = jvp(u_θ, (z, r, t), (v, 0, 1))",
    "u_tgt = v − (t − r)·dudt",
    "loss = adaptive_weight(‖u − sg(u_tgt)‖²)",
    "更新 θ"
  ],
  theory: R`恒等式是平均速度定义的直接推论，因此最优解就是真实平均速度，不需要人为设计一致性约束。训练目标不依赖老师、不需要课程。`,
  results: { head: ["模型", "NFE", "FID (ImageNet 256)"], rows: [
    ["MF-B/2", "1", "6.17"], ["MF-XL/2", "1", "3.43"], ["MF-XL/2", "2", "2.20"]
  ], note: "从头训练、240 epoch 量级。消融显示 r≠t 的比例、时间采样分布和 JVP 切向量的正确性都很关键。" },
  limits: R`JVP 带来额外算力与显存开销；目标方差较大；引导强度在训练时固定。后续 iMF、pMF、方差削减等工作分别针对这些问题。`,
  relations: R`$u(z_t,r,t)$ 就是 flow map 的平均速度参数化；在 Boffi 等的框架里属于 Eulerian 型自蒸馏。TVM 改为在终点处匹配，Drifting 等分布层面方法在 2026 年超过了它。`
},

boffi_cm: {
  src: "known",
  tldr: R`把"如何从头训练 flow map"系统化为自蒸馏：对角线 $X_{t,t}$ 用 flow matching 学速度，非对角线用 flow map 自身满足的方程监督自己。按所用方程分为 Eulerian、Lagrangian、Progressive 三类，统一了 CM、Shortcut、MeanFlow 等方法。`,
  motivation: R`各种一步/少步方法看起来五花八门，作者想说明它们其实都是在用流映射的不同数学刻画构造训练目标，并比较这些选择的优劣。`,
  method: R`
<p>参数化 $X_{s,t}(x)=x+(t-s)\,v_{s,t}(x)$。由于 $\lim_{s\to t}v_{s,t}=b_t$，对角线可以用插值的 flow matching 损失直接学：</p>
$$\mathcal L_{\text{diag}}=\mathbb E\,\|v_{t,t}(I_t)-\dot I_t\|^2.$$
<p>非对角线用三种自蒸馏之一：</p>
<ul>
<li><b>Eulerian（ESD）：</b>$\partial_s X_{s,t}(x)+\nabla X_{s,t}(x)\,v_{s,s}(x)=0$，需要空间雅可比–向量积；</li>
<li><b>Lagrangian（LSD）：</b>$\partial_t X_{s,t}(x)=v_{t,t}\big(X_{s,t}(x)\big)$，只需对时间求导；</li>
<li><b>Progressive（PSD）：</b>$X_{s,t}=X_{u,t}\circ X_{s,u}$，不需要导数。</li>
</ul>
<p>"自"蒸馏的含义是：方程里的速度 $v_{s,s}$ 来自同一个网络的对角线，而不是外部老师。</p>`,
  algo: [
    "采样 (s, t)，x_s = I_s(x0, x1)",
    "loss_diag = ‖v_θ(t,t, I_t) − İ_t‖²",
    "LSD: loss_off = ‖∂_t X_θ(s,t,x_s) − sg(v_θ(t,t, X_θ(s,t,x_s)))‖²",
    "ESD / PSD: 换成对应方程的残差",
    "loss = loss_diag + loss_off,  更新 θ"
  ],
  theory: R`证明了每种自蒸馏目标的极小点都是真实流映射，并讨论了不同目标在梯度方差、对空间导数的依赖上的差异。`,
  results: { head: ["发现", "说明"], rows: [
    ["方法比较", "在相同设置下系统比较三类目标；Eulerian 类依赖空间导数，训练信号方差更大"],
    ["统一性", "CM、CTM、Shortcut、MeanFlow 都可以写成某一类的特例"]
  ], note: "具体实验数字请以原文为准。" },
  limits: R`实验规模中等；三类方法的优劣在大规模、条件生成下是否保持，仍需验证。`,
  relations: R`MeanFlow 对应 Eulerian 型；Shortcut 对应 Progressive 型；TVM 在终点处匹配速度，与 Lagrangian 思想接近。建议作为"统一视角"的第一篇精读。`
},

ayf: {
  src: "known",
  tldr: R`NVIDIA 提出的连续时间 flow map 蒸馏方法：给出 Eulerian 和 Lagrangian 两种 flow map 蒸馏目标，在蒸馏中加入 autoguidance，再做短暂的对抗微调。蒸馏出的模型在少步和多步下质量都不退化。`,
  motivation: R`CM 一类模型在步数增加时质量反而下降，因为它们只学了"到终点"的映射。Flow map 可以支持任意步数，但缺少可扩展到大模型的蒸馏目标。`,
  method: R`
<p>学习 $f_\theta(x_t,t,s)\approx X_{t\to s}(x_t)$。提出两类连续时间损失（AYF-EMD 与 AYF-LMD），分别由 Eulerian 方程和 Lagrangian 方程在 $\Delta\to0$ 的极限得到，形式上与 sCM 的切线损失同构，并证明了它们推广了连续时间 CM 与 MeanFlow 类目标。</p>
<p><b>Autoguidance。</b>用一个较弱的模型来引导老师，作为蒸馏目标，从而避免 CFG 带来的过饱和。<b>对抗微调。</b>最后加一个短暂的对抗阶段提升锐度，同时尽量保持多样性。</p>`,
  algo: [
    "采样 t, s，x_t ← 加噪数据",
    "老师速度 v = autoguided_teacher(x_t, t)",
    "EMD/LMD 切线 g = 对 f_θ⁻ 做 JVP",
    "loss = ‖f_θ(x_t,t,s) − sg(f_θ⁻ − Δ·g)‖²",
    "(可选) 短暂对抗微调"
  ],
  theory: R`推导了两类连续时间 flow map 蒸馏目标，并说明它们包含连续时间一致性模型作为特例。`,
  results: { head: ["任务", "说明"], rows: [
    ["ImageNet 64 / 512", "少步生成达到当时同类方法的最好水平，且随步数增加质量不降"],
    ["文生图", "作者报告优于所有非对抗训练的少步采样器"]
  ], note: "摘要中还用解析和实验说明了 CM 随步数增加必然退化。具体数字请以原文为准。" },
  limits: R`仍需要强老师；对抗微调会牺牲一部分多样性。`,
  relations: R`和 Boffi 等的自蒸馏、MeanFlow 属同一数学框架，区别在于 AYF 用外部老师而不是自蒸馏。`
},

dmd: {
  src: "known",
  tldr: R`把一步生成器训练成"分布级"地与老师一致：用老师的 score 代表真实分布，用一个在线训练的 fake score 网络代表生成分布，两者之差给出最小化 KL 的梯度。`,
  motivation: R`逐点回归老师 ODE 的方法（如渐进蒸馏、CM）在一步时质量下降明显。作者认为没必要逐点复现老师的轨迹，只要生成分布对即可。`,
  method: R`
<p>一步生成器 $G_\theta(z)$ 产生 $x$，在各噪声水平加噪得 $x_t$。最小化扩散后的 KL：</p>
$$\nabla_\theta D_{\text{KL}}\approx\mathbb E_{t,z}\Big[w_t\,\big(s_{\text{fake}}(x_t,t)-s_{\text{real}}(x_t,t)\big)^\top\frac{\partial G_\theta(z)}{\partial\theta}\Big].$$
<p>其中 $s_{\text{real}}$ 是冻结的老师（带 CFG），$s_{\text{fake}}$ 是同步训练的扩散模型，用去噪损失拟合当前生成器的输出分布。另加一个回归损失：预先用老师 ODE 生成一批（噪声，图像）配对，用 LPIPS 让 $G_\theta$ 在这些噪声上复现老师的输出，以防模式坍缩。</p>`,
  algo: [
    "预先生成配对数据 {(z_ref, y_ref)} = 老师 ODE",
    "x = G_θ(z), t ~ U, x_t = 加噪(x, t)",
    "grad = w_t·(s_fake(x_t,t) − s_real(x_t,t))",
    "loss_G = ⟨sg(grad), x⟩ + λ·LPIPS(G_θ(z_ref), y_ref)",
    "loss_fake = 去噪损失(s_fake, sg(x))",
    "交替更新 G_θ 与 s_fake"
  ],
  theory: R`利用"扩散后两个分布 KL 的梯度可以写成两个 score 之差"这一事实；回归项则提供模式覆盖的约束。`,
  results: { head: ["任务", "NFE", "FID"], rows: [
    ["ImageNet 64", "1", "2.62"], ["COCO-30k 零样本 (SD 1.5)", "1", "11.49"]
  ], note: "文生图约 20 FPS。" },
  limits: R`需要预先用老师生成大量配对数据（代价高）；reverse KL 倾向于模式坍缩，生成图偏饱和；上限受老师限制。`,
  relations: R`和 Score Distillation Sampling（SDS）、VSD 同源。DMD2 去掉回归项并加入 GAN 损失。`
},

dmd2: {
  src: "known",
  tldr: R`DMD 的改进版：去掉昂贵的回归损失，用"双时间尺度更新"稳定 fake score，加入基于真实数据的 GAN 损失让学生可以超过老师，并通过"反向模拟"支持多步学生。`,
  motivation: R`DMD 的回归损失要预先生成上百万对样本，而且把学生质量锁死在老师之下。直接去掉回归项又会训练不稳定。`,
  method: R`
<ul>
<li><b>TTUR：</b>每更新一次生成器，就更新多次（如 5 次）fake score 网络，使其准确跟踪生成分布，从而在没有回归项时也能稳定训练；</li>
<li><b>GAN 损失：</b>在 fake score 网络的中间特征上接一个分类头，区分真实数据和生成样本。由于用的是真实数据，学生可以修正老师的误差；</li>
<li><b>多步生成器与反向模拟：</b>学生在固定的几个时间点上去噪。训练时的输入用学生自己生成的中间结果（而不是对真实图像加噪），消除训练/推理不一致。</li>
</ul>
$$\mathcal L_G=\mathcal L_{\text{DMD}}+\lambda_{\text{GAN}}\,\mathcal L_{\text{GAN}},\qquad \mathcal L_{\text{fake}}=\mathcal L_{\text{denoise}}+\lambda\,\mathcal L_{D}.$$`,
  algo: [
    "for each G step:",
    "  repeat 5×: 用 sg(G 输出) 更新 s_fake 的去噪损失 + 判别头",
    "  x = G_θ(z)（多步时用反向模拟得到输入）",
    "  loss_G = DMD 梯度项 + λ·GAN 生成器损失",
    "  更新 G_θ"
  ],
  theory: R`主要是经验与工程上的改进；GAN 项使目标不再完全受老师约束。`,
  results: { head: ["任务", "NFE", "FID"], rows: [
    ["ImageNet 64", "1", "1.28"], ["COCO 2014 零样本 (SD 1.5)", "1", "8.35"], ["SDXL", "4", "人评优于老师"]
  ], note: "ImageNet 64 上一步结果超过了老师。" },
  limits: R`对 GAN 超参数敏感；reverse KL 的多样性问题依然存在，后来的 Data-Forcing 等工作专门处理。`,
  relations: R`2025–2026 年大量视频/文生图加速工作（DMDR、AdvDMD、DyMD 等）都以 DMD2 为基线。`
},

imm: {
  src: "known",
  tldr: R`让模型学习从时间 $t$ 到任意更早时间 $s$ 的跳跃，并用"归纳"的方式训练：要求从 $t$ 和从稍早的 $r$ 出发跳到 $s$ 的两个分布一致，分布差用 MMD 在一组粒子上估计。单阶段从头训练，CM 是它的特例。`,
  motivation: R`CM 训练不稳定、需要课程。作者认为问题在于 CM 只做了逐样本的一阶矩匹配，统计效率低。如果改成真正的分布匹配，训练会更稳定。`,
  method: R`
<p>模型 $f_\theta(x_t,t,s)$ 把 $x_t$ 推到时间 $s\lt t$，记其推前分布为 $p^\theta_{s|t}$。对 $s\lt r\lt t$，用 DDIM 插值从 $x_t$ 得到 $x_r$，要求</p>
$$p^{\theta}_{s|t}\ \approx\ p^{\theta^-}_{s|r},$$
<p>用 MMD 衡量：对同一组中的 $M$ 个粒子计算</p>
$$\mathcal L=\text{MMD}^2\Big(\{f_\theta(x_t^{(i)},t,s)\}_{i=1}^M,\ \{f_{\theta^-}(x_r^{(i)},r,s)\}_{i=1}^M\Big),$$
<p>核函数用 Laplace 核等。$r$ 由 $(s,t)$ 决定并逐渐靠近 $t$，"归纳"的含义是：只要 $r$ 处已学好，$t$ 处就能通过匹配学好。</p>`,
  algo: [
    "x ~ data（按 M 个一组）, t ~ p(t), s < r < t",
    "x_t = 加噪(x, t);  x_r = DDIM(x_t, x, t → r)",
    "y_t = f_θ(x_t, t, s);  y_r = f_θ⁻(x_r, r, s)",
    "loss = 组内 MMD²(y_t, y_r)",
    "更新 θ, θ⁻ ← EMA 或 stop-grad"
  ],
  theory: R`证明了在合适条件下该目标的不动点对应正确的分布；证明 CM 等价于"单粒子、一阶矩"的 IMM，解释了 CM 不稳定的原因。`,
  results: { head: ["数据集", "NFE", "FID"], rows: [
    ["CIFAR-10", "2", "1.98"], ["ImageNet 256", "8", "1.99"]
  ], note: "均为从头单阶段训练。" },
  limits: R`一步质量不如多步；需要组内多个粒子，批大小和核函数选择会影响效果。`,
  relations: R`开启了"分布层面从头训练"这一分支，Drifting、W-Flow 都可以看作这一思路的延续。同团队（Luma）随后提出 TVM。`
},

rf: {
  src: "known",
  tldr: R`用直线插值做 flow matching（Rectified Flow），再用"reflow"：拿模型自己生成的（噪声，样本）配对重新训练，使轨迹越来越直，从而一步积分误差越来越小。`,
  motivation: R`ODE 轨迹弯曲是少步采样误差大的根源。如果能把噪声和数据之间的耦合改成"不交叉"的，轨迹就会变直。`,
  method: R`
<p>给定耦合 $(X_0,X_1)$，令 $X_t=tX_1+(1-t)X_0$，训练</p>
$$\min_v\int_0^1\mathbb E\,\|(X_1-X_0)-v(X_t,t)\|^2\,\mathrm dt.$$
<p>得到的 ODE $\mathrm dZ_t=v(Z_t,t)\mathrm dt$ 给出新的耦合 $(Z_0,Z_1)$。用新耦合再训练一次即为 reflow（k-rectified flow）。最后可以把直化后的流蒸馏成一步模型。</p>`,
  algo: [
    "1-RF: (X0, X1) 独立配对，训练 v",
    "生成配对: Z0 ~ N, Z1 = ODE_v(Z0)",
    "2-RF: 用 (Z0, Z1) 重新训练 v",
    "(可选) 蒸馏: 训练 G(Z0) ≈ Z1"
  ],
  theory: R`(1) Rectify 保持边缘分布不变；(2) 对任意凸代价，rectify 后的耦合传输代价不增；(3) 多次 reflow 使轨迹的"直度"收敛。`,
  results: { head: ["CIFAR-10 设置", "NFE", "FID"], rows: [
    ["1-RF", "1", "378"], ["2-RF", "1", "12.21"], ["2-RF + 蒸馏", "1", "4.85"]
  ], note: "说明直化后一步质量大幅提升。" },
  limits: R`每轮 reflow 都要生成大量配对，代价高；误差会随轮数累积；生成的配对本身有偏差。`,
  relations: R`InstaFlow 用它做文生图一步生成；OT 耦合、minibatch OT 是另一种拉直方式；MeanFlow 等方法则绕开了"拉直"，直接学弯曲轨迹的平均速度。`
},

instaflow: {
  src: "known",
  tldr: R`把 Rectified Flow 的 reflow 用在 Stable Diffusion 上：先 reflow 一次让轨迹变直，再蒸馏成一步生成器，得到当时最早的高质量一步文生图模型之一。`,
  motivation: R`直接把 SD 蒸馏成一步效果很差，因为原始耦合下轨迹弯曲、噪声到图像的映射很复杂。先拉直再蒸馏会容易得多。`,
  method: R`
<p>(1) 用 SD 1.5 作为初始化，生成（噪声，图像）配对，按 Rectified Flow 目标做文本条件的 reflow，得到 2-RF；(2) 在 2-RF 上用 LPIPS 等距离做一步蒸馏；(3) 扩大 U-Net（0.9B → 1.7B 的变体）进一步提升质量。</p>`,
  algo: [
    "配对: z ~ N, x = SD_ODE(z, prompt)",
    "Reflow: 用 (z, x) 训练 2-RF",
    "蒸馏: min LPIPS(G(z, prompt), 2-RF_ODE(z, prompt))"
  ],
  theory: R`沿用 Rectified Flow 的理论：reflow 降低传输代价、使轨迹变直，从而降低蒸馏难度。`,
  results: { head: ["模型", "数据", "FID"], rows: [
    ["InstaFlow-0.9B", "MS COCO 2017-5k", "23.3"], ["InstaFlow-1.7B", "MS COCO 2014-30k", "13.1"]
  ], note: "一步生成约 0.1 秒；训练约 199 A100 GPU 天。" },
  limits: R`一步图像在细节上明显弱于多步 SD；需要生成大量配对数据。后来被 DMD2、ADD 等方法超过。`,
  relations: R`是 ReFlow 分支在大模型上的代表应用。`
},

rf_improve: {
  src: "known",
  tldr: R`系统改进 Rectified Flow 的训练：指出一次 reflow 通常就够，并通过更合理的时间采样分布和距离函数，让 2-rectified flow 的一步质量大幅提升。`,
  motivation: R`普遍认为需要多轮 reflow 才能拉直，而每轮代价很高。作者分析发现，一次 reflow 后轨迹已经接近直线，剩余误差主要来自训练方式而不是轨迹曲率。`,
  method: R`
<ul>
<li>把时间采样改成两端更密的 U 形分布，因为 2-RF 的误差集中在 $t$ 接近 0 和 1 的位置；</li>
<li>用 LPIPS–Huber 之类的感知距离替代平方误差；</li>
<li>用 1-RF 的配对初始化 2-RF 的训练，以及其他训练细节改进。</li>
</ul>`,
  algo: [
    "用 1-RF 生成 (z, x) 配对",
    "t ~ U 形分布",
    "loss = LPIPS-Huber((x − z), v_θ(x_t, t))",
    "得到改进的 2-RF，1 步或 2 步采样"
  ],
  theory: R`论证了在一次 reflow 之后，耦合已经接近"不交叉"，从而解释为什么一次就够。`,
  results: { head: ["数据集", "说明"], rows: [
    ["CIFAR-10", "1-NFE FID 相比之前的 2-RF 最多降低 75%"],
    ["ImageNet 64", "一步、两步都优于一致性蒸馏和渐进蒸馏，FID 与 iCT 相当"]
  ], note: "" },
  limits: R`仍需要预训练流和生成配对的成本；依赖 LPIPS。`,
  relations: R`ReFlow 分支后期的代表工作；其中关于时间采样分布的经验也被后来的一步方法广泛借鉴。`
}

});

// ---- 以下依据 arXiv 原文整理 ----
Object.assign(window.PAPER_DETAILS, {

pmf: {
  src: "full",
  tldr: R`网络直接输出"去噪图像"$x$（低维流形上的量），再换算成平均速度 $u$ 并在速度空间用 MeanFlow 损失训练。这样在像素空间、无 VAE 的条件下实现一步生成：ImageNet 256 上 FID 2.22，512 上 2.48。`,
  motivation: R`现代生成模型有两个"拐杖"：多步采样和潜空间（VAE）。MeanFlow 去掉了多步，JiT 证明了像素空间 Transformer 用 $x$-预测可行，但两者还没结合。直接在像素空间预测 $u$ 会失败，因为 $u$ 是带噪的量，铺满整个高维空间，很难拟合。`,
  method: R`
<p><b>核心原则：</b>网络输出空间和损失空间分开设计。</p>
<p><b>去噪图像场。</b>定义 $x(z_t,r,t)\triangleq z_t-t\,u(z_t,r,t)$。当 $r=t$ 时它就是普通的去噪估计 $\mathbb E[x|z_t]$；当 $r=0$ 时它就是 ODE 终点（真实图像）；中间情形论文称为"广义流形假设"：$x$ 近似落在低维图像流形上。</p>
<p><b>参数化与损失。</b>网络输出 $x_\theta$，换算
$$u_\theta(z_t,r,t)=\frac1t\big(z_t-x_\theta(z_t,r,t)\big),$$
再沿用 iMF 的 $v$-loss：
$$\mathcal L_{\text{pMF}}=\mathbb E\,\|V_\theta-v\|^2,\qquad V_\theta\triangleq u_\theta+(t-r)\cdot\mathrm{JVP}_{\text{sg}}.$$
即"$x$-预测、$v$-损失"。可加感知损失（在 $x_\theta$ 上计算 LPIPS 类损失），并按 iMF 的方式支持 CFG。</p>`,
  algo: [
    "x ~ data, ε ~ N(0,I), (t, r) 采样, z = (1−t)x + tε, v = ε − x",
    "x_θ = net_θ(z, r, t)          # 网络直接输出图像",
    "u_θ = (z − x_θ) / t",
    "JVP = d u_θ / dt (沿 (v, 0, 1) 方向), stop-grad",
    "V_θ = u_θ + (t − r)·JVP",
    "loss = ‖V_θ − v‖² (+ 感知损失)",
    "采样: x = net_θ(ε, 0, 1)"
  ],
  theory: R`没有严格定理，核心是流形假设的经验验证：在 64×64、patch 维度 48 时 $x$-预测和 $u$-预测都可行（FID 3.80 vs 3.82）；在 256×256、patch 维度 768 时 $u$-预测崩溃（164.89），$x$-预测正常（9.56）。`,
  results: { head: ["设置", "NFE", "FID"], rows: [
    ["ImageNet 256，像素空间", "1", "2.22"], ["ImageNet 512，像素空间", "1", "2.48"],
    ["消融：u-预测 (256, B/16)", "1", "164.89"], ["消融：x-预测 (256, B/16)", "1", "9.56"],
    ["消融：加 EDM 式预条件", "1", "14.43（无预条件 3.53）"]
  ], note: "Muon 优化器收敛快于 Adam；感知损失有明显帮助；任何把网络输出变换为 x 的预条件器都会让输出偏离流形、显著变差。" },
  limits: R`仍依赖 JVP；与最好的多步潜空间模型（约 1.4）仍有差距；感知损失引入了预训练网络。`,
  relations: R`MeanFlow → iMF（$v$-loss）→ pMF（$x$-预测）一脉相承，也继承了 JiT"Back to Basics"的像素空间思路。`
},

tvm: {
  src: "full",
  tldr: R`学习任意两个时间之间的位移 $f_\theta(x_t,t,s)$，但不像 MeanFlow 那样在起点匹配，而是要求位移在终点 $s$ 处的时间导数等于终点处的真实速度。该目标给出 $W_2$ 距离上界；为满足所需的 Lipschitz 条件对 DiT 做了小改动。从头训练，ImageNet 256 上 1-NFE FID 3.29、4-NFE 1.99。`,
  motivation: R`直接监督位移需要在训练时积分 ODE，不可行。MeanFlow 等方法用"初始速度"约束，缺少与分布距离的理论联系。作者希望目标简单、可从头训练，并且有分布层面的保证。`,
  method: R`
<p>记净位移 $f(x_t,t,s)=\psi(x_t,t,s)-x_t=\int_t^s u(x_r,r)\,\mathrm dr$。参数化
$$f_\theta(x_t,t,s)=(s-t)\,F_\theta(x_t,t,s),\qquad u_\theta(x_t,t)=F_\theta(x_t,t,t),$$
同一个网络既是一步跳跃器，也是瞬时速度场。</p>
<p><b>终点速度条件。</b>真实位移满足 $\frac{\mathrm d}{\mathrm ds}f(x_t,t,s)=u(\psi(x_t,t,s),s)$。作者证明位移误差被终点速度误差的积分上界：
$$\mathcal L^t_{\text{displ}}\le\int_0^t\mathbb E\,\Big\|\tfrac{\mathrm d}{\mathrm ds}f_\theta(x_t,t,s)-u(\psi(x_t,t,s),s)\Big\|^2\mathrm ds.$$</p>
<p><b>用网络自身做代理。</b>把未知的 $\psi$ 和 $u$ 换成 $x_t+f_\theta(x_t,t,s)$ 和 $u_\theta$（带 stop-grad / EMA），得到
$$\mathcal L_{\text{TVM}}=\mathbb E\Big[\big\|\tfrac{\mathrm d}{\mathrm ds}f_\theta(x_t,t,s)-u_{\theta^-}\big(x_t+f_\theta(x_t,t,s),s\big)\big\|^2+\|u_\theta(x_s,s)-v_s\|^2\Big].$$
第二项是 flow matching；$t=s$ 时整个目标退化为 FM。$\frac{\mathrm d}{\mathrm ds}f_\theta$ 用 JVP 计算，作者写了支持 JVP 反向传播的融合注意力核。</p>
<p><b>Lipschitz 控制。</b>标准 DiT 训练中激活会爆炸。改用 RMSNorm 作 QK-Norm、把所有 LayerNorm 换成无参数 RMSNorm 等，使网络（半）Lipschitz。CFG 通过把目标速度换成 $w u(\cdot,c)+(1-w)u(\cdot)$ 训练进模型，不同 $w$ 是不同模型。</p>`,
  algo: [
    "x0 ~ data, ε, 采样 t ≥ s（一部分 t = s）",
    "x_t, x_s 由插值得到, v_s = 条件速度",
    "f, df/ds = JVP(f_θ(x_t,t,·), 对 s 求导)",
    "target = sg(u_θ⁻(x_t + f, s))",
    "loss = ‖df/ds − target‖² + ‖u_θ(x_s,s) − v_s‖²",
    "采样: x0 = x1 + f_θ(x1, 1, 0)，或多步"
  ],
  theory: R`(1) 终点速度误差上界位移误差；(2) 在模型 Lipschitz 的前提下，TVM 目标上界数据与模型分布的 2-Wasserstein 距离（差一个常数）；(3) $t=s$ 时退化为 flow matching。`,
  results: { head: ["数据集", "NFE", "FID"], rows: [
    ["ImageNet 256", "1", "3.29"], ["ImageNet 256", "4", "1.99"],
    ["ImageNet 512", "1", "4.32"], ["ImageNet 512", "4", "2.94"]
  ], note: "均为从头训练。消融：缩放参数化在 w=1.5 时很关键（1-NFE 6.04 vs 9.32）；训练中 t=s 的比例约 20% 时 2-NFE 最好。Luma 还给出了 10B+ 文生图的结果。" },
  limits: R`需要修改架构以满足 Lipschitz；JVP 带来工程负担；CFG 强度在训练时固定，且存在 FID 随训练在不同 $w$ 间此消彼长的现象。`,
  relations: R`与 MeanFlow 的区别在于匹配终点而非起点，思想上接近 Boffi 等的 Lagrangian 自蒸馏；由 IMM 的团队（Luma）提出。`
},

esc: {
  src: "full",
  tldr: R`为"从头训练的一步 shortcut 模型"（Shortcut、MeanFlow、IMM、sCT 等）建立统一设计框架，把理论推导和具体组件选择拆开，然后逐个改进组件。得到的 ESC 在 ImageNet 256 上一步 FID 2.85，训练步数加倍后为 2.53，无需预训练、蒸馏或课程。`,
  motivation: R`现有 shortcut 模型的理论推导与工程实现耦合在一起，看不清哪些设计真正起作用，难以做组件级改进。`,
  method: R`
<p><b>统一框架。</b>把各方法写成同一类连续时间 shortcut 一致性目标（论文记为 CTSC），区别只在于：速度目标怎么估计、时间怎么采样、用什么训练技巧等组件。</p>
<p><b>主要改进。</b></p>
<ul>
<li><b>Plug-in 速度：</b>用批内样本以自归一化权重 $w_q(y)\propto q(y)\,p(x_t|y)$ 估计边缘速度 $\mathbb E[v|x_t]$，替代单样本的条件速度，降低目标方差；论文证明了使用 plug-in 速度时 CTSC 损失的收敛性；</li>
<li><b>类别一致的批：</b>条件生成时，让同一批内的样本属于同一类别，使 plug-in 估计有意义；</li>
<li><b>渐进时间采样器</b>，以及引入 sCM 的训练技巧。</li>
</ul>`,
  algo: [
    "按类别组织 batch: {(x_i, ε_i)} 同类",
    "采样 (r, t)（渐进时间采样器）, z_t^i = 插值",
    "v_plugin(z_t^i) = Σ_j w_j(z_t^i)·v_j   # 自归一化权重",
    "以概率 p 用 v_plugin 替代条件速度",
    "按 MeanFlow 型目标计算 loss（含 sCM 技巧）",
    "更新 θ"
  ],
  theory: R`给出框架的有效性证明；证明 plug-in 速度下损失的收敛性（权重误差可被 $W_1$ 距离控制）。`,
  results: { head: ["设置", "NFE", "FID"], rows: [
    ["ImageNet 256，CFG", "1", "2.85"], ["ImageNet 256，2× 训练步数", "1", "2.53"],
    ["CIFAR-10", "1", "2.83（MeanFlow 2.92，IMM 3.20，sCT 2.97）"],
    ["小规模消融：MeanFlow 基线 → ESC", "1", "6.09 → 5.77"]
  ], note: "消融中每个组件的贡献都不大（0.1 量级），叠加后才明显。" },
  limits: R`单个改进幅度有限；plug-in 估计依赖批大小；作者在附录 F 讨论了局限。`,
  relations: R`适合当作"MeanFlow 系方法设计空间"的综述来读；和 MeanFlow 方差削减一文关注的是同一个问题（目标方差）。`
}

});

Object.assign(window.PAPER_DETAILS, {

soflow: {
  src: "full",
  tldr: R`直接学习速度 ODE 的"解函数" $f(x_t,t,s)$，用 flow matching 损失加一个不需要 JVP 的"解一致性"损失训练。同架构、同 epoch 下各尺寸都优于 MeanFlow（XL/2 一步 FID 2.96 vs 3.43）。`,
  motivation: R`MeanFlow 等方法依赖 JVP，在 PyTorch 中优化不好、显存和速度代价高；同时训练中缺少可直接用于 CFG 的速度估计。`,
  method: R`
<p><b>解函数。</b>在速度场 Lipschitz 的假设下，$f(x_t,t,s)$ 是从 $x_t$ 出发解 ODE 到时间 $s$ 的唯一解，满足 $f(x_t,t,t)=x_t$ 与 $\partial_3 f=v(f,s)$。</p>
<p><b>充分条件。</b>作者证明，只要模型满足
$$f_\theta(x_t,t,t)=x_t,\qquad \partial_1 f_\theta(x_t,t,s)\,v(x_t,t)+\partial_2 f_\theta(x_t,t,s)=0,$$
就有 $f_\theta=f$（第二式是 Eulerian 方程：沿轨迹移动起点，终点不变）。</p>
<p><b>无 JVP 的解一致性损失。</b>用有限差分近似上面的导数：取 $l\in(s,t)$ 接近 $t$，
$$\mathcal L_{\text{SCM}}=\mathbb E\Big[w\,\big\|f_\theta(x_t,t,s)-f_{\theta^-}\big(x_t+(\alpha'_t x_0+\beta'_t x_1)(l-t),\,l,\,s\big)\big\|^2\Big].$$
$l=t+(s-t)\,r(k,K)$，$r$ 随训练步数单调减小，使 $l$ 逐渐靠近 $t$（类似 CM 的课程）。权重含自适应项 $1/(\text{MSE}/(t-l)^2+\epsilon)^p$。</p>
<p><b>Flow matching 损失。</b>$s=t$ 处的导数即速度，用 FM 损失训练，这样训练中就有速度估计可用于 CFG。</p>`,
  algo: [
    "x0 ~ data, x1 ~ N(0,I), t, s ~ logit-normal (s < t)",
    "l = t + (s − t)·r(k, K)        # 随训练逐渐靠近 t",
    "x_t = α_t x0 + β_t x1,  v = α'_t x0 + β'_t x1",
    "L_SCM = w·‖f_θ(x_t,t,s) − sg f_θ(x_t + v(l−t), l, s)‖²",
    "L_FM  = FM 损失（提供 CFG 所需速度）",
    "更新 θ;  采样: x0 = f_θ(x1, 1, 0)"
  ],
  theory: R`证明了上述两个条件对 $f_\theta=f$ 是充分的（通过沿真实轨迹对中间时间求导为零得到）。`,
  results: { head: ["模型", "MeanFlow 1-NFE", "SoFlow 1-NFE"], rows: [
    ["B/2", "6.17", "4.85"], ["M/2", "5.01", "3.73"], ["L/2", "3.84", "3.20"], ["XL/2", "3.43", "2.96"]
  ], note: "ImageNet 256，FID-50K，相同 DiT 架构与 epoch。" },
  limits: R`有限差分引入 $l$ 的课程超参；仍在潜空间、仍需 CFG 训练技巧。`,
  relations: R`本质上是 Eulerian 型自蒸馏（同 MeanFlow）的有限差分版本，可以和 sCM/ECT 的"收紧 $\Delta t$"思路对照阅读。`
},

emf: {
  src: "full",
  tldr: R`从流映射的半群（轨迹一致性）出发，对短时间步做局部线性近似，把长程一致性目标改写成能直接用数据监督、且不需要 JVP 的形式（Euler Mean Flows）。同等 DiT-B 设置下，显存和单步耗时约减半，FID 也好于 MeanFlow。`,
  motivation: R`边界条件（$\phi_{t\to t}=\text{id}$，起点导数为速度）容易从数据监督，但轨迹一致性很难：半群约束有无穷多解；流映射也不存在与条件速度相容的"条件流映射"（论文给出证明）。Shortcut 一类靠局部向长程渐进扩展，误差累积；MeanFlow 用连续方程，需要 JVP。`,
  method: R`
<p>用平均速度 $u_{t\to r}(x)=\frac{\phi_{t\to r}(x)-x}{r-t}$ 改写半群关系：
$$(r-t)\,u_{t\to r}(x_t)=(s-t)\,u_{t\to s}(x_t)+(r-s)\,u_{s\to r}(x_s).$$
取 $s=t+\Delta t$（固定的小步长），由局部线性近似有 $u_{t\to s}\approx u_{t\to t}$，于是
$$u_{t\to r}(x_t)\approx u_{t\to t}(x_t)+(r-t-\Delta t)\,\frac{u_{t+\Delta t\to r}(x_{t+\Delta t})-u_{t\to r}(x_t)}{\Delta t}.$$
右边的瞬时速度 $u_{t\to t}$ 可以用数据的条件速度替代，从而得到直接由数据监督的长程目标，不需要 JVP。论文还给出 $x_1$-预测的变体。</p>`,
  algo: [
    "x0 ~ N, x1 ~ data, 采样 t < r, Δt 固定",
    "x_t = (1−t)x0 + t·x1,  x_{t+Δt} 同样由插值得到",
    "target = v_cond + (r−t−Δt)·sg(u_θ(x_{t+Δt}, t+Δt→r) − u_θ(x_t, t→r))/Δt",
    "loss = ‖u_θ(x_t, t→r) − target‖² / w(t,r)",
    "更新 θ"
  ],
  theory: R`在对流映射的合理假设（Assumption 1）下，定理 4.3 证明了该局部线性近似的有效性；并证明不存在自洽的条件流映射。`,
  results: { head: ["数据 (DiT-B 同等算力)", "方法", "峰值显存", "每步耗时", "FID"], rows: [
    ["CelebA-HQ", "MeanFlow", "32.1GB", "151.4ms", "12.4"], ["CelebA-HQ", "EMF", "23.3GB", "91.2ms", "10.9"],
    ["ImageNet", "MeanFlow", "101.9GB", "400.9ms", "11.1"], ["ImageNet", "EMF", "57.9GB", "198.8ms", "7.2"]
  ], note: "表格来自原文表 6；该表比较的是相同架构与算力下的训练目标。" },
  limits: R`作者把扩展到更大模型、更多任务和更一般的理论设定列为未来工作。`,
  relations: R`介于 Shortcut（离散半群）与 MeanFlow（连续方程）之间；和 SoFlow 一样追求"无 JVP"。`
},

otnfm: {
  src: "full",
  tldr: R`用 neural flow（直接参数化流映射的网络）学习噪声到数据的映射，实现无 ODE 的一步生成。指出朴素训练会出现"均值坍缩"，并证明一致的耦合是必要条件，因此用最优传输配对（minibatch / 在线耦合）解决。`,
  motivation: R`Flow matching 学速度、推理要积分。直接学流映射时，如果噪声与数据的配对是随机独立的，同一个噪声会对应很多数据，回归的最优解就是它们的均值，导致所有输出趋向数据均值。`,
  method: R`
<p>用 neural flow 参数化 $\phi_\theta(x_0,t)$，构造上满足 $\phi_\theta(x_0,0)=x_0$，直接回归插值路径上的点，不需要数值积分。</p>
<p><b>均值坍缩。</b>独立耦合下回归目标是条件期望，映射退化。作者证明：要学到非退化的映射，耦合必须是一致的（确定性的）。</p>
<p><b>OT 耦合。</b>用最优传输给出噪声–数据配对，并提出可扩展的 minibatch OT 与在线耦合策略。</p>`,
  algo: [
    "采样一批噪声 {x0} 与数据 {x1}",
    "求 minibatch OT 配对 π",
    "t ~ U, x_t = 插值(x0, x1_π)",
    "loss = ‖φ_θ(x0, t) − x_t‖²（或相应的路径匹配损失）",
    "推理: x1 = φ_θ(x0, 1)"
  ],
  theory: R`证明一致的耦合是非退化学习的必要条件，从理论上解释均值坍缩。`,
  results: { head: ["数据", "说明"], rows: [
    ["合成数据、MNIST、CIFAR-10", "一步生成得到有竞争力的样本质量，推理只需一次前向"]
  ], note: "实验规模较小，未在 ImageNet 级别验证。" },
  limits: R`OT 配对在高维、大数据上的可扩展性和偏差；实验规模有限。`,
  relations: R`把 ReFlow"耦合决定轨迹"的观点用到流映射学习上；与 Rectified Flow、minibatch OT flow matching 同源。`
}

});

Object.assign(window.PAPER_DETAILS, {

cumflow: {
  src: "full",
  tldr: R`把多步生成模型统一看成"瞬时流映射"（每一步的局部更新），再推广为"累积流映射"：直接学有限时间的转移。只需改时间嵌入和训练目标，就能把 EDM、DDIM、$x_1$-flow matching 等不同参数化变成少步甚至一步模型，主要面向图形学任务。`,
  motivation: R`MeanFlow 等方法绑定于 $u$-预测和图像生成，公式复杂，难以迁移到 EDM、DDIM、$x_1$-预测等图形学里常用的参数化上。`,
  method: R`
<p>把扩散、score matching、flow matching 统一为连续时间马尔可夫过程：模型学一个局部量 $m_t(x)$（如速度），通过短时转移核 $p_{t+h|t}(\cdot|x;m_t)$ 逐步采样，这就是"瞬时流映射"。</p>
<p><b>累积流映射。</b>把局部更新累积为有限时间转移，并给出与具体预测量无关的"累积参数化"，用于一步或少步生成。论文还讨论了训练累积参数化的困难和对应的解决办法。</p>
<p>实现上不改网络结构、不增加容量、不用蒸馏，只修改时间嵌入（加入第二个时间）和训练损失。</p>`,
  algo: [
    "选定基础模型参数化（EDM / DDIM / x1-FM / u-FM）",
    "时间嵌入增加目标时间 s",
    "按累积参数化构造 (t → s) 的训练目标",
    "推理: 以 1 到少数几步跨越 [t0, t1]"
  ],
  theory: R`给出瞬时流映射到累积流映射的统一抽象。`,
  results: { head: ["任务", "说明"], rows: [
    ["几何分布建模 (CFM-EDM)", "6 步，约 10× 加速且质量不降"], ["关节预测 (DDIM)", "10 步"],
    ["稀疏点条件 SDF 生成 (x1-FM)", "4 步"], ["图像条件草图生成 (DDIM)", "1 步"]
  ], note: "作者明确说明尚未在 ImageNet 等大规模数据上评估。" },
  limits: R`只评估了 5 个应用、4 种参数化；扩展到更大规模数据（如 ImageNet）是未来工作。`,
  relations: R`可视为把 MeanFlow / flow map 思想推广到非 $u$-预测参数化的工程框架。`
},

mf_var: {
  src: "full",
  tldr: R`从蒙特卡洛控制变量的角度解释 MeanFlow 为什么训练不稳定：条件速度在损失中扮演两个角色，作为回归目标没问题，但作为 JVP 切向量时会把噪声乘性放大。论文给出最优混合系数的闭式解，并把 iMF、AlphaFlow 等同期修补方法统一为对同一最优点的不同实现。`,
  motivation: R`MeanFlow 的损失不下降、梯度方差无界，大家已经提出多种补救，但缺少统一的统计解释。`,
  method: R`
<p>把条件速度分解为 $v_{\text{cond}}=v(x,t)+v'$，其中 $\mathbb E[v'|x_t]=0$，协方差 $\Sigma_{v'}$。代入 MeanFlow 损失得到
$$\mathbb E\big[\ell_{\text{MF}}\big]=\|r^{\text{sg}}_\theta\|^2+\text{sg}\big[\mathrm{Tr}(J\Sigma_{v'}J^\top)\big],\qquad J=(t-r)\,\partial_{x_t}u_\theta-I.$$
第一项是真正要最小化的残差；第二项来自切向量里的噪声，随 $J$ 的谱无界放大，是损失不降、梯度方差大的主因。</p>
<p><b>切向量混合。</b>把 JVP 的切向量换成
$$\tilde v_{\text{tang}}(\beta)=(1-\beta)\,v_{\text{cond}}+\beta\,\hat v,$$
$\hat v$ 是任意确定性的边缘速度代理（如速度网络），偏差 $b=\hat v-v$。$\beta=0$ 即原始 MeanFlow，$\beta=1$ 完全用代理。梯度中噪声项为 $((1-\beta)J-\beta I)v'$，偏差项为 $\beta(J+I)b$，从而有偏差–方差权衡，并可求出最优 $\beta^\ast$ 的闭式解。</p>`,
  algo: [
    "与 MeanFlow 相同地采样 (x, ε, t, r)",
    "v_cond = ε − x;  v_hat = 速度代理网络(z, t)",
    "tangent = (1 − β)·v_cond + β·v_hat",
    "dudt = JVP(u_θ, (z, r, t), (tangent, 0, 1))",
    "loss = ‖u_θ − sg(v_cond − (t − r)·dudt)‖²"
  ],
  theory: R`(1) 损失分解定理，说明方差来源；(2) 最优切向量混合系数的闭式解（各向同性近似下）；(3) 证明 iMF 的边缘速度头、AlphaFlow 的课程等是同一最优点的不同近似。`,
  results: { head: ["实验", "发现"], rows: [
    ["2D 基准", "按 β 扫描，复现了理论预测的偏差–方差顺序"],
    ["潜空间 DiT", "梯度 MSE 最小点在 β≈0.94，但 FID 最优在 β=0（无偏端）"],
    ["FID 与 MSE", "FID 对偏差的惩罚约为梯度 MSE 的 2.6 倍（超线性）"]
  ], note: "结论：方差最优不等于生成质量最优。" },
  limits: R`各向同性近似；FID 与梯度 MSE 的"景观错配"尚未完全解释。`,
  relations: R`解释了 iMF、AlphaFlow、Re-MeanFlow、TVM 等补救方法为什么有效；适合在研究 MeanFlow 训练稳定性前精读。`
},

decm: {
  src: "full",
  tldr: R`分析连续时间 CM 的两大问题：训练不稳定来自自监督项的发散，采样不灵活来自误差累积。从整个 $(t,s)$ 子轨迹空间中只挑三类关键子轨迹优化：一致性轨迹（蒸馏）、瞬时轨迹（flow matching 作边界正则），以及新的"噪声到带噪点"(N2N) 映射。ImageNet 256 上一步 FID 1.70。`,
  motivation: R`CM 的损失可以分解为"全监督项"和"自监督项"，后者梯度更不稳定；且过度强调一致性会冲掉预训练模型的瞬时速度概念。同时多步采样的误差在第一步就开始累积。`,
  method: R`
<p>用双时间参数化 $f_\theta(x_t,t,s)=x_t+(s-t)F_\theta(x_t,t,s)$。不优化全部 $O(n^2)$ 子轨迹，只选三类：</p>
<ul>
<li><b>一致性轨迹</b>（$s=1$）：连续时间 CM 蒸馏目标，实现少步生成；</li>
<li><b>瞬时轨迹</b>（$s=t$）：flow matching 损失，作为边界正则稳定训练；</li>
<li><b>N2N 轨迹</b>：从噪声映射到任意中间带噪点，减轻第一步的误差累积，使采样更灵活。</li>
</ul>`,
  algo: [
    "从老师得到速度 v_tea(x_t, t)",
    "L_cm  = 连续时间 CM 目标 (s = 1)",
    "L_fm  = ‖F_θ(x_t, t, t) − v‖²",
    "L_n2n = 噪声 → 任意 x_s 的映射目标",
    "loss = L_cm + L_fm + L_n2n"
  ],
  theory: R`对 CM 损失做分解，经验上展示自监督项的梯度不稳定；对误差累积给出分析。`,
  results: { head: ["方法 (ImageNet 256)", "NFE", "FID"], rows: [
    ["DE-CM", "1", "1.70"], ["DE-CM", "2", "1.33"], ["DE-CM", "50", "1.26"],
    ["FACM†（作者复现）", "1", "1.76"], ["sCM†（作者复现）", "1 / 2", "3.34 / 1.94"]
  ], note: "† 为作者在自己配置下的蒸馏复现（250 epoch）。还展示了 2–50 步的文生图结果。" },
  limits: R`属于蒸馏方法，依赖老师；三类子轨迹的选择偏经验。`,
  relations: R`与 FACM 同样把 flow matching 当"锚"；N2N 映射与 flow map 的任意起止点思想一致。`
},

facm: {
  src: "full",
  tldr: R`认为连续时间 CM 不稳定是因为只训练"捷径"目标会让网络遗忘瞬时速度场，于是加入 flow matching 任务作为"锚"，两个任务用不同的条件信号区分。蒸馏 LightningDiT 后，ImageNet 256 上一步 FID 1.70、两步 1.32；并用 Chain-JVP 扩展到 14B 的 Wan 2.2。`,
  motivation: R`sCM 等需要专门的架构改动来稳定训练。作者把问题归因于"缺少锚点"：CM 目标本身没错，但单独训练时网络会忘掉定义整个流的瞬时速度。`,
  method: R`
<p><b>CM 学的就是平均速度。</b>在 OT-FM 参数化 $f_\theta=x_t+(1-t)F_\theta$ 下，理想的 $F_\theta$ 是到终点的平均速度 $\bar v$，满足
$$\bar v(x_t,t)=v(x_t,t)+(1-t)\frac{\mathrm d\bar v(x_t,t)}{\mathrm dt},$$
与 MeanFlow 恒等式（$r\equiv1$）相同。</p>
<p><b>FACM 目标。</b>$\mathcal L_{\text{FACM}}=\mathcal L_{\text{FM}}+\mathcal L_{\text{CM}}$。FM 是"锚"，回归瞬时速度 $v=v_{\text{base}}+w(v_{\text{cond}}-v_{\text{uncond}})$（从头训练时 $v_{\text{base}}=x_1-x_0$，蒸馏时来自冻结的老师）；CM 是"加速器"。两任务用不同的条件信号（扩展的时间区间）区分，使同一个网络能解耦地同时学两件事，不需要改架构。</p>
<p><b>Chain-JVP。</b>节省显存、与 FSDP 兼容的 JVP 实现，用于 14B 模型。</p>`,
  algo: [
    "x_t = 插值, 老师 / 数据给出 v（含 CFG）",
    "L_FM = ‖F_θ(x_t, c_FM) − v‖²",
    "dF/dt = Chain-JVP(F_θ⁻, x_t, t)",
    "L_CM = 连续时间 CM 目标（条件 c_CM）",
    "loss = L_FM + L_CM"
  ],
  theory: R`推导出 CM 目标等价于学习平均速度场，说明 FM 锚定与 CM 之间的关系。`,
  results: { head: ["设置", "NFE", "FID"], rows: [
    ["ImageNet 256（蒸馏 LightningDiT）", "1", "1.70"], ["ImageNet 256", "2", "1.32"],
    ["CIFAR-10 无条件", "1", "2.69（MeanFlow 2.92，sCT 2.85）"],
    ["Wan 2.2 14B 文生图", "2–8", "原为 2×40 步"]
  ], note: "" },
  limits: R`大模型上 1–2 步与 8 步之间仍有质量差距；Chain-JVP 的计算开销仍较大。`,
  relations: R`与 DE-CM、SoFlow 一样用 FM 损失稳定训练；把 CM 与 MeanFlow 的关系讲得很清楚，适合对照阅读。`
}

});

Object.assign(window.PAPER_DETAILS, {

drifting: {
  src: "full",
  tldr: R`换一种看问题的方式：扩散模型在推理时迭代地推动样本，Drifting 则把这个演化放在训练时完成。定义一个"漂移场"$V_{p,q}$，生成分布与数据分布相等时漂移为零；训练目标是让网络输出向"输出 + 漂移"靠拢。推理天然一步。ImageNet 256 上 FID 1.54（潜空间）/ 1.61（像素空间）。`,
  motivation: R`生成建模就是找一个映射 $f$，使推前分布 $f_\#p_\epsilon\approx p_{\text{data}}$。神经网络训练本身就是迭代的：每一步参数更新都会让样本移动一点 $\Delta x_i=f_{i+1}(\epsilon)-f_i(\epsilon)$。既然如此，可以直接设计这个"训练时的移动"，而不是在推理时迭代。`,
  method: R`
<p><b>漂移场与平衡。</b>要求漂移场反对称：$V_{p,q}(x)=-V_{q,p}(x)$，于是 $q=p\Rightarrow V=0$（反过来一般不成立，论文对核化形式给出 $V\approx0\Rightarrow q\approx p$ 的充分条件）。</p>
<p><b>训练目标（不动点迭代）。</b>
$$\mathcal L=\mathbb E_\epsilon\Big[\big\|f_\theta(\epsilon)-\text{sg}\big(f_\theta(\epsilon)+V_{p,q_\theta}(f_\theta(\epsilon))\big)\big\|^2\Big],$$
损失的数值就是 $\mathbb E\|V\|^2$。</p>
<p><b>吸引与排斥。</b>受 mean-shift 启发：
$$V^+_p(x)=\tfrac1{Z_p}\mathbb E_p\big[k(x,y^+)(y^+-x)\big],\quad V^-_q(x)=\tfrac1{Z_q}\mathbb E_q\big[k(x,y^-)(y^--x)\big],\quad V=V^+_p-V^-_q.$$
正样本 $y^+$ 来自真实数据（吸引），负样本 $y^-$ 来自当前生成样本（排斥），用批内样本估计。</p>
<p><b>特征空间漂移。</b>在预训练编码器（论文用 latent-MAE 等）的特征空间里计算漂移，并可在多尺度、多位置的特征上求和：$\sum_j\|\phi_j(x)-\text{sg}(\phi_j(x)+V(\phi_j(x)))\|^2$。编码器只在训练时用。</p>`,
  algo: [
    "ε ~ p_ε;  x = f_θ(ε)              # 一批生成样本 (负样本)",
    "y⁺ ~ 真实数据 (同类)               # 正样本",
    "在特征空间计算 V = V⁺(φ(x)) − V⁻(φ(x))",
    "loss = Σ_j ‖φ_j(x) − sg(φ_j(x) + V_j)‖²",
    "更新 θ;  推理: x = f_θ(ε)，一步"
  ],
  theory: R`反对称性保证平衡点包含 $q=p$；对核化漂移场给出 $V\approx0$ 推出 $q\approx p$ 的充分条件（附录 C.1）。`,
  results: { head: ["设置", "FID"], rows: [
    ["ImageNet 256，潜空间，1-NFE", "1.54"], ["ImageNet 256，像素空间，1-NFE", "1.61"],
    ["消融（B/2，100 ep）：反对称 V⁺−V⁻", "8.46"], ["消融：1.5× 吸引", "41.05"], ["消融：只吸引 V⁺", "177.14"]
  ], note: "消融说明吸引与排斥必须严格平衡（反对称）。" },
  limits: R`依赖预训练特征编码器；需要较大的批内正负样本；损失景观与 FID 的关系仍不清楚。`,
  relations: R`与 GAN 都是"推前分布匹配"，但不用判别器；与 MMD/IMM 关系密切（核函数驱动的吸引–排斥）。W-Flow、Priming 都直接在它的基础上改进。`
},

wflow: {
  src: "full",
  tldr: R`先在分布空间规定一条从参考分布到数据分布的 Wasserstein 梯度流（能量取 Sinkhorn 散度），再训练一个静态的一步生成器把这条演化"压缩"进去。ImageNet 256 上一步 FID 1.29，比 FID 相近的多步扩散模型快约 100 倍。`,
  motivation: R`Drifting 的漂移场是启发式设计的。作者希望把"训练动力学"建立在有原则的基础上：既保证收敛到目标分布，又能在高维中高效实现、覆盖所有模式。`,
  method: R`
<p><b>增量构造。</b>生成器写成局部传输的复合：$f^{(k+1)}=g^{(k)}\circ f^{(k)}$，$g^{(k)}(x)=x+\eta V^{(k)}(x)$。为保持一步推理，每一步都训练网络 $f_{\theta^{(k+1)}}$ 去拟合 $g^{(k)}\circ f_{\theta^{(k)}}$（回归到粒子更新后的位置）。</p>
<p><b>能量泛函的选择。</b>速度场取能量的 Wasserstein 梯度：</p>
<ul>
<li>MMD：$V=\int\nabla k(x,y)p(dy)-\int\nabla k(x,y)q(dy)$，分布相距较远时核值很小，梯度消失；</li>
<li>KL：$V=\nabla\log p-\nabla\log q$，需要估计 score，有偏差、不稳定；</li>
<li>Sinkhorn 散度：$V=T^\varepsilon_{q,p}(x)-T^\varepsilon_{q,q}(x)$，基于熵正则 OT 的映射，只用样本即可计算，能感知全局分布差异。</li>
</ul>
<p>引导通过"速度引导"项支持 CFG。</p>`,
  algo: [
    "z ~ p_ref;  x = f_θ(z)",
    "y ~ 数据 batch",
    "T_qp = 熵 OT(x → y) 的重心映射;  T_qq = 熵 OT(x → x)",
    "V = T_qp(x) − T_qq(x)            # Sinkhorn 散度的 WGF 速度",
    "loss = ‖f_θ(z) − sg(x + η·V)‖²",
    "更新 θ;  推理: x = f_θ(z)"
  ],
  theory: R`证明在适当假设下，有限样本的训练动力学收敛到连续时间的分布动力学；引用已有结果说明高斯假设下 Sinkhorn 散度的 WGF 收敛到全局最小。`,
  results: { head: ["设置", "FID (1-NFE)"], rows: [
    ["ImageNet 256，XL", "1.29"], ["ImageNet 256，L", "1.35"],
    ["散度消融：Drifting 漂移场", "8.46"], ["散度消融：MMD", "10.40"], ["散度消融：KL", "10.17"], ["散度消融：Sinkhorn", "7.29"]
  ], note: "还报告了更好的模式覆盖和域迁移能力。" },
  limits: R`熵 OT 的计算依赖批大小与正则强度；作者把其它能量泛函和文生图列为未来工作。`,
  relations: R`直接把 Drifting 的漂移场换成有原则的 WGF 速度；与 DSB / 薛定谔桥共享熵正则 OT 的工具。`
},

priming: {
  src: "full",
  tldr: R`两阶段训练一步生成器：先做简单的重建预训练（像素空间、固定噪声水平 $s=0.8$ 的去噪），再在特征空间做分布层面的后训练（OT 目标）。核心发现：只要有这个重建阶段，分布训练从头开始的困难就基本消失了。ImageNet 256 上一步 FID 1.19，算力约为同等 FID 方法的十分之一。`,
  motivation: R`直接从随机初始化做分布层面的训练（如 Drifting）很难、很贵；而一致性、flow map、分布蒸馏等方法又需要复杂的机制。是不是想得太复杂了？`,
  method: R`
<p><b>第一阶段：重建预训练。</b>固定噪声水平 $s=0.8$，让网络从 $x_s$ 预测干净图像，并加感知损失：
$$\mathcal L=\mathbb E\big[\|f_\theta(x_s,c)-x_0\|^2+\lambda\|\phi(f_\theta(x_s,c))-\phi(x_0)\|^2\big],$$
$\phi$ 为 LPIPS（$\lambda=0.4$）和 ConvNeXt（$\lambda=0.1$）。</p>
<p><b>第二阶段：分布后训练。</b>把输入换成纯噪声，在 ResNet-MAE 和 ConvNeXt 特征空间里用熵正则 OT 距离做分布匹配：$\mathcal D=0.75\,\text{OT}_{0.001}+0.25\,\text{OT}_{0.01}$。每个类别在线生成 128 个样本，其中 64 个用作 stop-gradient 项，共 25,000 次更新。不使用 CFG。</p>
<p><b>为什么有效：Jacobian 对齐。</b>一步梯度更新引起的特征变化为 $\Delta h_\theta\approx-\eta J_\theta J_\theta^\top v$。预训练越久，损失梯度越与 $J_\theta J_\theta^\top$ 的强响应方向对齐，后训练效率越高。</p>
<p><b>多步扩展。</b>后训练时在一组噪声水平上都训练"预测干净图像"，推理时交替预测和重新加噪（类似 CM 的多步采样）。</p>`,
  algo: [
    "# 阶段 1：重建",
    "x_s = (1−s)x0 + s·ε,  s = 0.8",
    "loss = ‖f_θ(x_s,c) − x0‖² + λ·‖φ(f_θ(x_s,c)) − φ(x0)‖²",
    "# 阶段 2：分布后训练",
    "x = f_θ(ε, c)  (每类在线生成一组)",
    "loss = OT_ε(φ(x), φ(x_real)) 的组合（含 stop-grad 项）"
  ],
  theory: R`没有收敛定理；用 Jacobian 对齐的实验分析解释重建预训练为何让分布训练变容易。`,
  results: { head: ["设置", "FID", "算力 (EFLOPs)"], rows: [
    ["Priming-L（160 ep 预训练）", "1.19", "238.95"], ["Priming-L（80 ep）", "1.27", "151.29"],
    ["附录：Priming-L + FD-loss 后训练（160 ep）", "0.94", "—"]
  ], note: "像素空间、一步、无 CFG。附录中改用 FD-loss（直接匹配特征均值和协方差）后训练时 FID 低于 1，但该损失直接优化与 FID 相近的统计量。文生图实验基于 MiniT2I（MM-JiT-B/16），与 MeanFlow 基线比较 GenEval 和 DPG-Bench；还做了世界模型实验。" },
  limits: R`依赖预训练特征编码器，存在"直接优化评测指标"的嫌疑（作者自己也提出了这一问题）；固定噪声水平等选择较经验。`,
  relations: R`继承 Drifting 的特征空间分布训练，但用重建预训练替代复杂机制；标题本身就是对 flow map 路线的直接质疑。`
},

diffratio: {
  src: "full",
  tldr: R`不需要老师监督的一步扩散训练：VSD/DMD 用两个 score 网络之差来估计分布匹配的梯度，这里改为直接训练一个分类器估计"生成分布 / 数据分布"在各噪声水平上的对数密度比，其梯度就是 score 差。预训练扩散模型只用于初始化生成器，不提供监督。`,
  motivation: R`分数蒸馏有两种偏差来源：老师 score 的误差，以及学生 score 的估计误差。而且需要两个大的辅助网络。（注：本文 arXiv 初版标题为 "Towards Training One-Step Diffusion Models Without Distillation"。）`,
  method: R`
<p>扩散 KL 的梯度依赖 score 差，而它恰好是对数密度比的梯度：
$$\nabla_{x_t}\log q_\theta(x_t)-\nabla_{x_t}\log p_d(x_t)=\nabla_{x_t}\log\frac{q_\theta(x_t)}{p_d(x_t)}.$$
训练一个以 $t$ 为条件的二分类器 $c_\eta(x_t,t)$ 区分真实（$y=1$）与生成（$y=0$）样本，则
$$\nabla_{x_t}\log\frac{q_\theta}{p_d}=\nabla_{x_t}\,\text{logit}\big(1-c_\eta(x_t,t)\big),$$
代入得到生成器的梯度 $\sum_t w(t)\,\mathbb E\big[\nabla_{x_t}\text{logit}(1-c_\eta)\,\partial x_t/\partial\theta\big]$。这把分数分布匹配和 GAN 式对抗训练联系了起来。</p>
<p>学到的密度比还可以当作"验证器"，用于推理时的并行扩展（多采几个样本选最好的），不需要外部奖励。</p>`,
  algo: [
    "生成器 G_θ 用预训练扩散模型初始化",
    "x = G_θ(z);  对 x 和真实数据都在随机 t 加噪",
    "训练 c_η(x_t, t)：真实→1，生成→0",
    "g = ∇_{x_t} logit(1 − c_η(x_t, t))",
    "生成器: loss = ⟨sg(w(t)·g), x_t⟩",
    "推理时可用 c_η 给多个样本打分择优"
  ],
  theory: R`推导了密度比梯度与 score 差的等价性；分析并实验验证该估计器的梯度偏差小于 VSD（2D 例子上 log-MMD −6.08 vs −5.25）。`,
  results: { head: ["数据集", "说明"], rows: [
    ["CIFAR-10、ImageNet 64、ImageNet 512", "一步生成优于大多数需要老师的蒸馏方法"],
    ["2D 高斯混合", "log-MMD −6.08（VSD −5.25），log 密度 −6.79（VSD −10.88）"]
  ], note: "具体 FID 请对照原文表格。" },
  limits: R`仍依赖预训练扩散模型初始化；分类器训练与 GAN 一样可能不稳定。`,
  relations: R`介于 DMD（score 差）和 GAN（判别器）之间，可以把它看作两者的桥梁。`
},

rcm: {
  src: "full",
  tldr: R`把 sCM 首次扩展到 10B+ 文生图 / 文生视频，发现它在细节上有缺陷，于是加入 DMD 风格的 score 蒸馏作为"长跳"正则（rCM = sCM + λ·DMD）。在 Cosmos-Predict2、Wan2.1（最大 14B）上 1–4 步生成，质量接近 DMD2，多样性更好。`,
  motivation: R`sCM 的前向一致性目标会把小时间处的误差传播到大时间处，且前向散度偏向覆盖模式，导致细节（文字、精细纹理）差；而 DMD 类方法质量高但易模式坍缩。另外大模型上 JVP 难以实现。`,
  method: R`
<p><b>工程：</b>实现了支持并行（FSDP / 上下文并行）的 FlashAttention-2 JVP 核，使 sCM 能在 10B+ 模型和视频上训练；提出稳定时间导数计算的技巧。</p>
<p><b>Score 正则：</b>
$$\mathcal L_{\text{DMD}}=\mathbb E\Big[\big\|x_0-\text{sg}\big[x_0-\tfrac{f_{\text{fake}}(x_t,t)-f_{\text{teacher}}(x_t,t)}{\text{mean}(|x_0-f_{\text{teacher}}(x_t,t)|)}\big]\big\|^2\Big],\qquad \mathcal L_{\text{rCM}}=\mathcal L_{\text{sCM}}+\lambda\,\mathcal L_{\text{DMD}},$$
$\lambda=0.01$ 对所有模型和任务都适用。DMD 提供反向（mode-seeking）散度，与 sCM 的前向目标互补。</p>
<p><b>Rollout：</b>学生样本由随机步数 $N\in[1,N_{\max}]$ 的"去噪–加噪"交替生成，时间点从 $p_D$ 随机抽取并保证单调递减，只对最后一步反传 DMD 梯度。</p>`,
  algo: [
    "L_sCM: 连续时间一致性蒸馏（JVP 核）",
    "x0 = 学生随机 N 步 rollout 的结果",
    "x_t = 加噪(x0, t),  t ~ p_D",
    "L_DMD 按上式（归一化的 fake − teacher 差）",
    "loss = L_sCM + 0.01·L_DMD;  同时训练 fake score"
  ],
  theory: R`从散度角度解释：sCM 的前向目标误差从小时间传向大时间；反向散度作为长跳正则可以修复。`,
  results: { head: ["模型/任务", "步数", "说明"], rows: [
    ["Cosmos-Predict2、Wan2.1（≤14B），T2I / T2V", "1–4", "质量指标上一般与 DMD2 相当，多样性更好"],
    ["加速比", "—", "15×–50×"]
  ], note: "作者指出，大规模扩散蒸馏的最好方法仍是 DMD2 一类的 score/对抗蒸馏，rCM 的目标是在保持质量的同时缓解模式坍缩。" },
  limits: R`仍需老师；JVP 工程复杂；需要 fake score 网络。`,
  relations: R`sCM（分支 1）与 DMD（分支 3）的结合，说明"前向 + 反向散度"互补是当前大模型蒸馏的一个方向。`
}

});

Object.assign(window.PAPER_DETAILS, {

ddmd: {
  src: "full",
  tldr: R`把实际使用的 DMD 梯度拆成两项：分布匹配项（DM）和此前被忽视的"CFG 增强"项（CA）。实验表明在文生图这类需要 CFG 的任务里，真正驱动"多步变少步"的是 CA，DM 主要起正则、稳定和抑制伪影的作用。据此为两项分别设计重新加噪的时间表。`,
  motivation: R`大家普遍认为 DMD 有效是因为"学生分布匹配老师分布"。但理论上的 DMD 梯度（用条件 score）效果很差，实际有效的版本把真实 score 换成了 CFG score，这个改动的作用一直没人细究。`,
  method: R`
<p>实际 DMD 梯度用 $s^{\text{real}}_{\text{cfg}}=s_{\text{uncond}}+\alpha(s_{\text{cond}}-s_{\text{uncond}})$，代入后可分解为
$$\nabla_\theta\mathcal L_{\text{DMD}}=\mathbb E\Big[-\Big(\underbrace{s^{\text{real}}_{\text{cond}}-s^{\text{fake}}_{\text{cond}}}_{\text{DM}}+(\alpha-1)\underbrace{\big(s^{\text{real}}_{\text{cond}}-s^{\text{real}}_{\text{uncond}}\big)}_{\text{CA}}\Big)\frac{\partial G_\theta(z_t)}{\partial\theta}\Big].$$</p>
<p><b>消融：</b>分别只用 CA、只用 DM、两者都用。结论是 CA 是"矛"，负责少步转换；DM 是"盾"，负责稳定。</p>
<p><b>机制分析：</b>CA 的效果取决于重新加噪的时间 $\tau$：只在高噪声区间计算时主要增强低频（色块、构图），扩展到较干净的时间后才加入高频细节。DM 则通过 fake 模型持续跟踪学生分布，纠正 CA 单独使用时产生的伪影。</p>
<p><b>改进：</b>CA 与 DM 使用解耦的重新加噪时间表。</p>`,
  algo: [
    "x = G_θ(z_t)（少步时用 backward simulation）",
    "τ_CA ~ CA 的时间表;  τ_DM ~ DM 的时间表",
    "CA = (α−1)(s_cond^real − s_uncond^real)(x_{τ_CA})",
    "DM = (s_cond^real − s_cond^fake)(x_{τ_DM})",
    "loss_G = ⟨−sg(CA + DM), x⟩;  fake 模型照常用去噪损失跟踪"
  ],
  theory: R`主要是对梯度的代数分解与控制实验，没有收敛性定理。`,
  results: { head: ["实验", "说明"], rows: [
    ["文生图（与 LCM、Turbo 等比较）", "FID、CLIP、ImageReward、HPS 等指标上有竞争力"],
    ["工业应用", "DMDR 论文中提到 Z-Image-Turbo 通过 Decoupled-DMD 蒸馏得到"]
  ], note: "" },
  limits: R`结论主要针对需要 CFG 的条件生成；无 CFG 场景下两项的作用未必相同。`,
  relations: R`AMD 把它纳入"禁区"优化框架；对理解为什么 DMD 一类方法会过饱和（CA 本身就是一种外推）很有帮助。`
},

dmdr: {
  src: "full",
  tldr: R`把 DMD 蒸馏和强化学习放在一个框架里联合优化：RL 让蒸馏"偏向偏好"而非均匀压缩整个分布，DMD 反过来作为正则抑制奖励投机（reward hacking）。少步学生在多个指标上超过多步老师。`,
  motivation: R`蒸馏与 RL 通常是先后独立的两个后训练阶段，"先蒸馏再 RL"容易奖励投机，且受老师上限约束。`,
  method: R`
<p>DMD 梯度为 $-\mathbb E_t\big[(s_{\text{real}}(F_t)-s_{\text{fake}}(F_t))\,\mathrm dG_\theta/\mathrm d\theta\big]$。DMDR 分两个阶段：</p>
<ul>
<li><b>阶段 1：</b>带奖励倾斜的分布匹配（Reward-Tilted Distribution Matching），以及两种动态蒸馏训练策略；</li>
<li><b>阶段 2：</b>DMD 与 RL（默认 ReFL，奖励模型为 DFN-CLIP 和 HPSv2.1）联合优化，DMD 项把学生拉在老师分布附近，防止投机。</li>
</ul>`,
  algo: [
    "阶段 1：奖励倾斜的 DMD + 动态蒸馏策略（约 1.5K 步）",
    "阶段 2：loss = L_DMD + λ·L_RL(ReFL)（约 1.5K 步）",
    "fake score 网络照常跟踪学生分布"
  ],
  theory: R`从 KL 正则 RL 的角度说明 DMD 可以作为 RL 的分布正则。`,
  results: { head: ["SDXL", "NFE", "HP Score"], rows: [
    ["Base（老师）", "50", "27.15"], ["DMD2", "1", "26.60"], ["DMDR", "1", "32.01"],
    ["DMD2", "4", "28.57"], ["DMD2 + PSO（先蒸馏后 RL）", "4", "29.87"], ["DMDR", "4", "32.98"]
  ], note: "也在 SD3-Medium、SD3.5-Large 上验证，并报告 CLIP、Aesthetic、PickScore、GenEval、DPG-Bench。" },
  limits: R`依赖奖励模型；指标多为偏好类分数，与真实多样性之间可能存在取舍。`,
  relations: R`R_dm、AdvDMD 都在"DMD + RL 统一"这条线上继续推进。`
},

amd: {
  src: "full",
  tldr: R`从优化角度重看 DMD：学生样本在潜空间里被"老师吸引、fake 模型排斥"。在"禁区"里，真实老师给出的引导不可靠、fake 模型的排斥力又太弱，训练就会出问题。AMD 用奖励代理检测禁区，动态调整两种梯度的权重，并让 fake 模型更关注失败样本（排斥景观锐化）。`,
  motivation: R`DMD 的稳定性问题缺少统一的解释，各种改进（GAN 项、回归项、加大噪声、老师自适应）看起来互不相关。`,
  method: R`
<p><b>优化视角。</b>一阶近似下，参数更新等价于对样本 $x=G_\theta(z)$ 做梯度下降 $x\leftarrow x-\eta_{\text{eff}}\nabla_x\mathcal V_{\text{DMD}}$，其中
$$\mathcal V_{\text{DMD}}(x)=\tfrac12\|x-\hat x_{0,\text{real}}\|^2-\tfrac12\|x-\hat x_{0,\text{fake}}\|^2=\tfrac12\|d_{\text{real}}\|^2-\tfrac12\|d_{\text{fake}}\|^2.$$
第一项是吸引，第二项是排斥。</p>
<p><b>统一分类。</b>DMD（回归项作外力）、DMD2（判别器外力）、Decoupled-DMD（噪声重置）、DMDR（奖励外力）等都被归为避开禁区的隐式策略。</p>
<p><b>AMD。</b>(1) 结构化信号分解，按奖励优势动态调整 CA 与 DM 的系数 $\alpha d_{ca}+\beta d_{dm}$；(2) 排斥景观锐化：fake 模型的去噪损失按优势加权
$$\mathcal L_\psi=\mathbb E\big[W(\tilde a)\cdot\|\epsilon-\psi(x_t,t)\|^2\big],\quad W(\tilde a)=e^{-\tilde a},$$
让 fake 模型更认真地学会"失败样本长什么样"，从而产生更强的排斥。</p>`,
  algo: [
    "x = G_θ(z);  用奖励代理算优势 ã",
    "d_ca, d_dm 由老师 / fake 模型得到",
    "g = α(ã)·d_ca + β(ã)·d_dm",
    "生成器按 g 更新",
    "fake: loss = e^{−ã}·‖ε − ψ(x_t,t)‖²"
  ],
  theory: R`命题 3.1 给出参数更新与潜空间势能下降的等价（一阶近似）。`,
  results: { head: ["任务", "说明"], rows: [
    ["SDXL", "HPSv2 从 30.64 提升到 31.25"], ["Wan2.1 等视频模型", "在 VBench、GenEval 上提升保真度和训练鲁棒性"]
  ], note: "" },
  limits: R`依赖奖励代理检测禁区；权重函数的形式较经验。`,
  relations: R`适合当作 DMD 系改进方法的"综述框架"来读，表 1 把六种方法放在同一个算子里比较。`
},

rdm: {
  src: "full",
  tldr: R`把分布匹配本身看成一种"奖励" $R_{\text{dm}}$，从而用 RL 的工具来做蒸馏：组归一化（GNDM）让估计更稳定，可以自适应地与外部奖励结合，还能用重要性采样提升样本效率。`,
  motivation: R`已有的"蒸馏 + RL"方法只是把两个损失相加，算法上并没有统一。`,
  method: R`
<p>DMD 的目标是最小化学生与老师分布的反向 KL，梯度为 $-\mathbb E[(s_{\text{real}}-s_{\text{fake}})\nabla_\theta G_\theta]$。作者把这个信号重写为每个样本的奖励 $R_{\text{dm}}$，于是：</p>
<ul>
<li><b>GNDM：</b>借鉴 GRPO 的组归一化，用同组样本的均值统计量给出更稳健的优化方向；</li>
<li><b>奖励融合：</b>与外部奖励模型自然地加权组合；</li>
<li><b>重要性采样：</b>按 RL 方式复用样本，提高效率。</li>
</ul>`,
  algo: [
    "对每个 prompt 生成一组样本（SDE backward simulation）",
    "R_dm(x) = 由 s_real − s_fake 得到的逐样本奖励",
    "A = 组内归一化(R_dm + λ·R_ext)",
    "按带重要性采样的策略梯度更新 G_θ"
  ],
  theory: R`在算法层面建立 DMD 与 RL 的对应关系。`,
  results: { head: ["实验", "说明"], rows: [
    ["与 vanilla DMD 比较", "GNDM 更稳定、效果更好"], ["与外部奖励结合", "自适应权重优于简单相加"]
  ], note: "具体数字请对照原文表格。" },
  limits: R`奖励化后仍然依赖 fake score 的估计质量。`,
  relations: R`和 DMDR、AdvDMD 同属"蒸馏与 RL 统一"方向，区别在于它把 DMD 本身奖励化。`
},

advdmd: {
  src: "full",
  tldr: R`把 DMD2 里对抗训练的判别器直接当作奖励模型，放进 GRPO 框架里做 RL：真实图高分、生成图低分，在线与学生一起更新，并对去噪过程的中间状态也打分，从而缓解奖励投机。4 步 SD3.5 在 DPG-Bench 上超过 40 步原模型。`,
  motivation: R`现有"蒸馏 + RL"是组合式的，需要外部奖励模型，容易被投机；而 DMD2 里本来就有一个会随学生更新的判别器。`,
  method: R`
<p>回顾 DMD 与 Flow-GRPO 后：</p>
<ul>
<li><b>对抗奖励：</b>判别器在真实与生成样本上对抗训练，作为 GRPO 的奖励；对中间状态和最终状态都训练，监督整条采样轨迹；</li>
<li><b>统一目标：</b>DMD 损失与 RL 损失联合优化；</li>
<li><b>训练策略：</b>统一的 SDE backward simulation（同时给 DMD 和 GRPO 提供带随机性的轨迹），以及 DMD 与 RL 不同的更新节奏。</li>
</ul>`,
  algo: [
    "SDE backward simulation 生成一组轨迹",
    "r = D(中间/最终状态)        # 对抗判别器作为奖励",
    "A = 组归一化(r)，按 GRPO 更新学生",
    "同时计算 DMD 梯度（各自的更新节奏）",
    "D 用真实 vs 生成样本在线更新"
  ],
  theory: R`以经验设计为主。`,
  results: { head: ["模型", "步数", "说明"], rows: [
    ["SD3.5", "4", "DPG-Bench 超过原始 40 步模型"], ["SD3", "4", "GenEval 显著提升"], ["Qwen-Image", "2", "优于对比方法"]
  ], note: "" },
  limits: R`判别器奖励本身可能被利用；GAN 训练的不稳定性仍在。`,
  relations: R`与 DMDR 相比不需要外部奖励模型；与 DMD2 共用判别器。`
},

dfd: {
  src: "full",
  tldr: R`对 DMD2 做"一行代码"的修改：以概率 $p$ 让老师去给一段真实视频打分，而不是给学生的生成结果打分，更新方向变为 $s_{\text{fake}}-s_{\text{real}}$。这样抵消了反向 KL 的模式寻求倾向，同时恢复了被坍缩的模式（多样性）和过饱和的偏离流形样本（保真度）。`,
  motivation: R`DMD 最小化反向 KL，天然 mode-seeking：视频蒸馏后多样性大幅下降（例如相机运动几乎固定），画面过饱和。`,
  method: R`
<p>DMD2 的更新是在学生样本 $G_\theta(z)$ 上计算 $s_{\text{real}}-s_{\text{fake}}$。DFD 以概率 $p$ 把这个点换成真实视频 $x$，此时更新方向变为 $g_\theta\propto s_{\text{fake}}-s_{\text{real}}$（在真实数据点上评估）。直觉上，这相当于引入一个前向 KL 风格的项：哪里有真实数据但学生覆盖不足，就把学生往那里推。</p>`,
  algo: [
    "以概率 1−p:  x̃ = G_θ(z)          # 原 DMD2",
    "以概率 p:    x̃ = 真实视频 x       # Data-Forcing",
    "x̃_t = 加噪(x̃, t)",
    "按 DMD2 计算 s_real, s_fake 并更新（方向随样本来源变化）"
  ],
  theory: R`从反向 KL 的 mode-seeking 性质出发解释其失败模式，并说明真实数据项起到的平衡作用。`,
  results: { head: ["方法", "VBench 平均", "DINO 多样性", "相机轨迹距离"], rows: [
    ["老师", "0.899", "0.301", "26.65"], ["DMD2", "0.901", "0.190", "4.28"],
    ["DP-DMD", "0.903", "0.197", "3.47"], ["DFD", "0.906", "0.205", "19.26"]
  ], note: "相机运动多样性几乎恢复到老师水平，同时美学分数最高。" },
  limits: R`需要真实视频数据；$p$ 需要调节。`,
  relations: R`是缓解 DMD 多样性问题的最简单方案之一，可与 rCM（加前向散度）对照。`
},

dymd: {
  src: "full",
  tldr: R`面向机器人视频世界模型的 DMD：直接蒸馏会抑制"机器人–物体"之间的运动。作者发现原因是弱加噪让老师后验集中在运动不足的样本附近，而运动强的样本 fake score 拟合误差更大。DyMD 让老师监督和 critic 拟合都随学生自适应调整。`,
  motivation: R`DMD 蒸馏后的视频世界模型画面质量不错，但物体交互运动明显减弱，这对具身预测是致命的。`,
  method: R`
<ul>
<li><b>时间亲和度条件的重加噪采样：</b>根据每个 rollout 当前的交互保真度，把基础时间表与一个"老师先验"（由局部后验变化启发）混合，在恢复运动与精修外观之间平衡；</li>
<li><b>动态引导的 fake score 跟踪：</b>用一个以噪声为条件的预测器估计每个 rollout 的相对拟合难度，在 critic 损失中加大"难"样本（通常是运动强的样本）的权重。</li>
</ul>`,
  algo: [
    "rollout = G_θ(z)，计算其时间亲和度（交互保真度）",
    "t ~ 混合(基础时间表, 老师先验 | 亲和度)",
    "按 DMD 计算学生梯度",
    "critic: loss = w(难度预测)·去噪损失"
  ],
  theory: R`对老师后验与 fake score 误差做了分析，说明运动被抑制的机制。`,
  results: { head: ["设置", "说明"], rows: [
    ["14B 老师 → 1.3B 四步学生", "任务一致性优于基础 DMD，视觉质量相当"]
  ], note: "" },
  limits: R`针对具身/世界模型场景设计，通用性有待验证。`,
  relations: R`和 Data-Forcing 一样关注 DMD 在视频上的"动态/多样性"损失，但从时间表和 critic 加权入手。`
}

});

Object.assign(window.PAPER_DETAILS, {

apt: {
  src: "full",
  tldr: R`扩散预训练之后，直接对真实数据做对抗训练（Adversarial Post-Training），把 8B 的文生视频 MMDiT 变成一步生成器：实时生成 2 秒、1280×720、24fps 的视频，也能一步生成 1024px 图像。`,
  motivation: R`图像领域的蒸馏方法一步时质量下降明显，视频更难。作者认为不必以老师为目标，而应像 LLM 的后训练一样，直接针对真实数据训练。`,
  method: R`
<p><b>目标：</b>非饱和 GAN 损失
$$\mathcal L_D=\mathbb E[f_D(D(x,c))]+\mathbb E[f_G(D(G(z,c),c))],\quad \mathcal L_G=\mathbb E[g_G(D(G(z,c),c))],$$
$f_D=g_G=\log\sigma$，$f_G=\log(1-\sigma)$。</p>
<p><b>生成器初始化：</b>直接对扩散模型做对抗训练会坍缩，所以先用 MSE 一致性蒸馏（CFG 7.5）得到一个虽然模糊但可用的一步模型，再做对抗训练。</p>
<p><b>判别器：</b>与扩散模型同架构，用扩散权重初始化，在多层上接输出头得到标量 logit。</p>
<p><b>近似 R1 正则：</b>用 $\|D(x)-D(x+\epsilon\eta)\|^2$ 近似梯度惩罚，避免二阶导。没有它判别器损失迅速归零，生成器只输出色块。</p>`,
  algo: [
    "阶段 1：一致性蒸馏（MSE，CFG 7.5）得到 Ĝ 作为初始化",
    "阶段 2：交替训练",
    "  D: min L_D + λ·‖D(x) − D(x + σξ)‖²（近似 R1）",
    "  G: min L_G",
    "推理: x = z − G(z, c, T)，一步"
  ],
  theory: R`以经验为主。`,
  results: { head: ["任务", "说明"], rows: [
    ["文生视频（8B）", "1 步实时生成 2 秒 720p 24fps 视频"], ["文生图", "1 步 1024px，质量与当时最好的方法相当"]
  ], note: "作者也指出一步生成在结构和文本对齐上仍有退化。" },
  limits: R`结构、文本对齐退化；对抗训练对正则和初始化很敏感。`,
  relations: R`开启了"扩散预训练 + 对抗后训练"的路线，SeedVR2 和 Adversarial Flow Models 都出自同一团队（字节跳动 Seed）。`
},

seedvr2: {
  src: "full",
  tldr: R`把 APT 用到一步视频修复（超分）：提出随输出分辨率自适应的窗口注意力，以及一组稳定对抗训练的损失（含高效的特征匹配损失），一步达到或超过多步方法。`,
  motivation: R`扩散视频修复质量好但太慢；把一步蒸馏方法扩展到高分辨率真实视频修复非常困难。`,
  method: R`
<p>沿用 APT 的两阶段：确定性蒸馏 → 对抗后训练。针对视频修复的改进：</p>
<ul>
<li><b>自适应窗口注意力：</b>窗口大小随输出分辨率动态调整，避免高分辨率下固定窗口带来的不一致；</li>
<li><b>损失增强：</b>验证了一系列稳定对抗训练的损失，并提出不明显增加训练开销的特征匹配损失。</li>
</ul>`,
  algo: [
    "输入低质视频，条件化生成器",
    "阶段 1：确定性蒸馏",
    "阶段 2：对抗后训练 + 特征匹配等损失",
    "推理：一步输出高分辨率视频"
  ],
  theory: R`以经验为主。`,
  results: { head: ["任务", "说明"], rows: [["真实世界视频修复", "一步达到或优于现有多步方法"]], note: "" },
  limits: R`主要面向修复任务；对抗训练仍需较多工程技巧。`,
  relations: R`APT 的应用扩展。`
},

afm: {
  src: "full",
  tldr: R`对抗训练的生成器加一个最优传输正则 $\mathbb E\|G(z)-z\|^2$，使它学到唯一的确定性 OT 映射（与 flow matching 的映射一致），而不是任意一个传输映射，从而大幅稳定 GAN 训练。原生支持一步和多步，ImageNet 256 上 XL/2 一步 FID 2.38，并能端到端训练 56 层、112 层的模型。`,
  motivation: R`传统 GAN 的生成器可以学到任意一个把噪声推到数据的映射，训练中会"漂移"、不稳定。一致性类方法则必须学好中间时间步才能传播，占用模型容量、累积误差。`,
  method: R`
<p><b>对抗目标：</b>相对论式 GAN 损失 $f(a,b)=-\log\sigma(a-b)$，并加 R1、R2 梯度惩罚，用有限差分近似：$\mathcal L_{r1}\approx\mathbb E\big[\tfrac1{\epsilon^2}\|D(x)-D(\mathcal N(x,\epsilon^2I))\|^2\big]$，$\epsilon=0.01$，只在 25% 样本上计算。</p>
<p><b>OT 正则：</b>由 Brenier 定理，高斯源分布、二次代价下最优传输映射唯一。令 $z$ 与 $x$ 同维，加
$$\mathcal L^G_{\text{ot}}=\mathbb E_z\Big[\tfrac1n\|G(z)-z\|^2\Big],$$
在对抗目标保证边缘分布匹配的前提下，把解偏向唯一的 OT 映射 $G^\ast$。对抗梯度归一化到单位尺度后，$\lambda_{\text{ot}}$ 可以跨模型大小通用。</p>
<p><b>多步与引导：</b>同一框架支持多步生成，并讨论了分类器引导和 CFG 的接入方式。</p>`,
  algo: [
    "z ~ N(0, I) (与数据同维), x ~ data",
    "D: relativistic loss + 近似 R1/R2",
    "G: L_adv + λ_ot·‖G(z) − z‖²/n",
    "推理: x = G(z)，一步（或多步）"
  ],
  theory: R`基于 Brenier 定理说明 OT 正则在全局最优处选出唯一的确定性映射。`,
  results: { head: ["设置", "NFE", "FID"], rows: [
    ["ImageNet 256，XL/2", "1", "2.38"], ["ImageNet 256，B/2", "1", "接近一致性方法的 XL/2"]
  ], note: "还展示了不借助中间监督端到端训练 56 层和 112 层模型。" },
  limits: R`仍需判别器与梯度惩罚；需要噪声与数据同维。`,
  relations: R`把 ReFlow / flow matching 的"确定性 OT 映射"与 GAN 结合，是对抗分支的代表作；与 APT 同一团队。`
},

nextgan: {
  src: "full",
  tldr: R`常规 GAN 只监督判别器的输出值，不直接规定生成器拿到的梯度。NextGAN 让标量 critic 的输入梯度去回归"真实特征与生成特征之差"（在冻结表征空间的插值点上），对固定生成器，这等价于带梯度能量惩罚的对抗目标。同一个目标可以把最大 14B 的预训练图像/视频生成器改造成一步或少步模型。`,
  motivation: R`GAN 的梯度信号是间接的、难以控制；现有少步方法又往往需要辅助损失、在线老师 score 或轨迹。`,
  method: R`
<p>独立配对真实样本 $\bar x_r$ 与生成样本 $\bar x_p=g_\theta(z,c)$，在冻结表征 $f$ 中得到 $x_r,x_p$，位移 $\Delta=x_r-x_p$，插值点 $x_\alpha$。critic $u_\phi$ 的目标：
$$\mathcal L_{\text{field}}=\tfrac12\,\mathbb E\big[\|\nabla_x u_\phi(x_\alpha,c)-\Delta\|^2\big].$$
回归的结果是"总体失配场" $v_\theta(x,c)=\mathbb E[X_r-X_p\mid X_\alpha=x,c]$（在线性势函数类上是其保守投影）。生成器沿 $\nabla u_\phi$ 做标准的势函数型 GAN 更新。</p>`,
  algo: [
    "x_r ~ data, x_p = g_θ(z, c)，映射到冻结特征",
    "Δ = x_r − x_p;  x_α = (1−α)x_p + α·x_r",
    "critic: min ½‖∇u_φ(x_α, c) − Δ‖²",
    "生成器: 沿 ∇u_φ(f(g_θ(z,c))) 方向更新（势函数 GAN 更新）"
  ],
  theory: R`证明对固定生成器，该回归目标等价于带梯度能量惩罚的对抗目标；回归恢复失配场或其保守投影（命题 1）。`,
  results: { head: ["设置", "NFE", "结果"], rows: [
    ["ImageNet 256，像素 JiT-B", "1", "FID 2.87"], ["ImageNet 256，潜空间 DiT-XL/1", "1", "FID 1.54"],
    ["对抗形式", "1", "FID 1.36"], ["Wan2.1-1.3B 文生视频", "2", "VBench 83.13（比 rCM 高 1.25）"]
  ], note: "单个模型同时支持 1、2、4 步采样。" },
  limits: R`依赖冻结表征空间；独立随机配对带来的方差。`,
  relations: R`与 Drifting / W-Flow 一样在特征空间做分布匹配，但用"学一个 critic 的梯度场"替代批内核估计；可视为把 GAN 和漂移场统一起来。`
}

});

Object.assign(window.PAPER_DETAILS, {

fmlm: {
  src: "full",
  tldr: R`在 one-hot token 嵌入上做连续流（FLM），质量追平离散扩散；连续形式下存在唯一的 flow map，可以直接蒸馏成少步模型（FMLM），其一步生成质量超过近期少步离散扩散语言模型的 8 步结果。挑战了"离散数据必须用离散加噪"的普遍看法。`,
  motivation: R`离散扩散语言模型少步时质量急剧下降，难以真正提速。作者指出，离散方法不具备唯一的 flow map 这一结构，而连续流具备。`,
  method: R`
<p><b>FLM：</b>在 one-hot 嵌入（单纯形）上定义连续流。流本身和它的 flow map 都可以用尊重单纯形几何的交叉熵目标来学习，而不是 MSE。</p>
<p><b>FMLM：</b>把 FLM 蒸馏为 flow map。论文识别出三种不同的 flow map 蒸馏方式并做了实际比较。</p>`,
  algo: [
    "x1 = one-hot(tokens), x0 ~ N, x_t = 插值",
    "FLM: 用交叉熵预测 x1（等价于学速度）",
    "FMLM: 以 FLM 为老师，蒸馏 X_{s→t}（三种目标之一）",
    "推理: 1 步或少步，最后取 argmax"
  ],
  theory: R`论证连续形式下存在唯一 flow map，而离散加噪过程没有对应结构。`,
  results: { head: ["设置", "结果"], rows: [
    ["FLM（LM1B、OWT）", "与最好的离散扩散基线相当（如 MDLM 的 Gen. PPL 109.21 / 105.15）"],
    ["FMLM 一步", "超过少步离散扩散语言模型 8 步的质量"]
  ], note: "评测用 Gen. PPL 和熵（防止以降低多样性换取 PPL）。" },
  limits: R`规模仍是 LM1B/OWT 级别；连续表示在大词表、大模型上的效率待验证。`,
  relations: R`把图像领域的 flow map 蒸馏（Boffi 等、AYF）直接搬到语言上。`
},

didi: {
  src: "full",
  tldr: R`从预训练的扩散大语言模型（掩码扩散）初始化，蒸馏出少步学生。理论基础是"积分 KL 散度"最小化，并借助策略梯度绕开离散状态不可微的问题。在 OpenWebText 上 8 NFE 困惑度 62.2、128 NFE 为 18.4，优于之前的加速方法和 GPT-2，最高 64× 加速。`,
  motivation: R`连续空间的分布匹配蒸馏需要对 $z_t$ 求导，而掩码扩散的前向过程包含 argmax 等不可微操作。`,
  method: R`
<p><b>DiDi-Instruct：</b>把积分 KL（IKL，即对所有噪声水平的 KL 积分）扩展到掩码扩散模型；用策略梯度的形式得到严格的梯度估计，不需要对离散采样求导。</p>
<p><b>稳定性与质量：</b>分组奖励归一化、中间状态匹配（提升覆盖），以及奖励引导的祖先采样器（提升推理质量）。</p>`,
  algo: [
    "学生 ← 预训练 dLLM 初始化",
    "学生少步生成序列，按掩码过程得到中间状态",
    "用判别 / 密度比信号构造奖励，组内归一化",
    "按策略梯度最小化 IKL 更新学生",
    "推理: 奖励引导的祖先采样"
  ],
  theory: R`提出基于积分 KL 最小化的框架，并给出离散情形下的可行梯度。`,
  results: { head: ["OpenWebText", "NFE", "困惑度"], rows: [
    ["DiDi-Instruct", "8", "62.2"], ["DiDi-Instruct", "128", "18.4"]
  ], note: "熵损失约 1%；相比其它 dLLM 蒸馏基线，训练时间减少 20 倍以上。还验证了蛋白质序列生成。" },
  limits: R`主要在 GPT-2 规模上验证。`,
  relations: R`相当于离散版的 DMD / Diff-Instruct。`
},

t3d: {
  src: "full",
  tldr: R`扩散大语言模型减少解码步数时，因为同一步并行解出的 token 被假设相互独立（分解误差）而质量骤降。T3D 让少步学生匹配全步老师的生成轨迹（轨迹自蒸馏），并加入直接判别优化（DDO，一种反向 KL、mode-seeking 的目标），在推理和代码任务上效果更好。`,
  motivation: R`掩码扩散语言模型的少步解码受"token 分解误差"限制。`,
  method: R`
<p>在掩码扩散语言模型（MDLM）上：</p>
<ul>
<li><b>轨迹自蒸馏：</b>用全步老师（即模型自身多步解码）生成的轨迹作为监督，训练少步学生；理论与实验都表明轨迹级监督能缓解分解误差；</li>
<li><b>DDO：</b>反向 KL 目标，让学生偏向老师的主要模式，对困难推理任务更有利。</li>
</ul>`,
  algo: [
    "老师（全步）生成轨迹 x_1 → ... → x_0",
    "学生按少步时间表在轨迹状态上预测",
    "loss = 轨迹匹配损失 + DDO 反向 KL 项"
  ],
  theory: R`分析轨迹级监督如何减小分解误差。`,
  results: { head: ["任务", "说明"], rows: [["推理、代码生成基准", "少步解码质量明显优于基线"]], note: "" },
  limits: R`依赖老师轨迹；mode-seeking 目标可能牺牲多样性。`,
  relations: R`与图像里的 CM / 渐进蒸馏思想相近，DDO 部分与 DMD 的反向 KL 对应。`
},

coupling: {
  src: "full",
  tldr: R`一步离散生成的根本困难是表示能力：并行地给每个位置独立输出分布，只能表示分解的序列分布。Coupling Model 先构造离散序列与高斯隐变量之间的连续耦合（用归一化流把隐变量边缘对齐到 $\mathcal N(0,I)$），再训练一个并行解码器反演这个耦合，推理时采样 $z$、一次解码全部 token。`,
  motivation: R`不用自回归、也不用多步细化时，token 之间的依赖无从表达；而把多步离散采样器蒸馏成少步，又受老师和复杂单纯形流的限制。`,
  method: R`
<p>解码器在给定共享隐变量 $z$ 时分解：$G_\theta(x|z)=\prod_t G_\theta(x_t|z)$，但边缘 $p(x)=\int G_\theta(x|z)p_Z(z)\,dz$ 是隐变量混合，可以携带全局依赖。</p>
<ul>
<li><b>阶段 A（序列 → 高斯）：</b>把每个训练序列映射成连续表示，用似然训练的归一化流使诱导的隐变量边缘对齐 $\mathcal N(0,I)$，得到成对监督 $(z,x)$；</li>
<li><b>阶段 B（反演）：</b>训练并行解码器从 $z$ 解码 $x$。</li>
</ul>
<p>与 flow map 压缩不同，它不沿着单纯形轨迹学终点映射，而是先构造一个可采样的高斯耦合再学一步逆映射。</p>`,
  algo: [
    "阶段 A: x → 连续表示 h(x) → 归一化流 → z，最大似然使 z ~ N(0,I)",
    "得到配对 (z, x)",
    "阶段 B: 训练 G_θ(x | z) = Π_t G_θ(x_t | z)，交叉熵",
    "推理: z ~ N(0,I)，一次前向并行解码"
  ],
  theory: R`附录 C 形式化了逐 token 直接一步解码的表达能力障碍。`,
  results: { head: ["任务", "相对最强一步基线"], rows: [
    ["LM1B 文本生成（困惑度）", "降低 33%"], ["果蝇增强子设计（FBD）", "降低 18%"], ["MNIST-Binary（FID）", "降低 46%"]
  ], note: "结论：一步离散生成的效果很大程度上取决于解码前数据与噪声如何耦合。" },
  limits: R`两阶段训练；归一化流的表达能力限制耦合质量。`,
  relations: R`"耦合决定一切"的观点与 ReFlow、OT-NFM 一致，但用于离散数据。`
},

mfm: {
  src: "full",
  tldr: R`把 flow map / 一致性模型推广到随机情形：Meta Flow Map 能从任意中间状态 $x_t$ 一步采样出任意多个独立的干净样本 $x_1\sim p_{1|t}(\cdot|x_t)$。这些样本可重参数化、可微，从而高效估计价值函数，用于推理时引导和微调两种奖励对齐。`,
  motivation: R`奖励对齐（无论推理时引导还是微调）的最优解都需要价值函数，而价值函数依赖后验 $p_{1|t}$，通常只能靠昂贵的轨迹模拟来估计。`,
  method: R`
<p>目标是从奖励倾斜分布 $p_{\text{reward}}(x)\propto p_1(x)e^{r(x)}$ 采样。由 Doob $h$-变换，最优受控动力学需要价值函数
$$V_t(x_t)=\log\mathbb E_{x_1\sim p_{1|t}(\cdot|x_t)}\big[e^{r(x_1)}\big].$$
MFM 学一个随机的一步后验采样器，从 $x_t$ 直接生成 $p_{1|t}$ 的独立样本，于是 $V_t$ 及其梯度可以用蒙特卡洛加重参数化高效估计，无需模拟轨迹。</p>`,
  algo: [
    "训练 MFM: 给定 x_t 与额外噪声 ξ，输出 x1 ~ p_{1|t}(·|x_t)",
    "对齐时: 采样 K 个 x1^(k) = MFM(x_t, ξ_k)",
    "V_t ≈ log mean_k exp(r(x1^(k)))，可微",
    "推理时引导: 用 ∇V_t 修正速度; 或用于微调目标"
  ],
  theory: R`从 Doob $h$-变换出发统一推理时引导和微调，说明一步后验采样足以给出价值函数估计。`,
  results: { head: ["任务", "说明"], rows: [
    ["类别条件 / 文生图引导", "展示了基础 MFM 与引导后 MFM 的样本对比"],
    ["逆问题、黑盒奖励", "同一框架可以处理"]
  ], note: "具体数值请对照原文。" },
  limits: R`需要额外训练随机后验采样器；后验近似误差会传到价值估计。`,
  relations: R`把 CM / flow map 从"确定性一步"扩展为"随机一步后验"，是 flow map 用于对齐的代表作。`
},

vfm: {
  src: "full",
  tldr: R`Flow map 没有显式采样轨迹，难以加入外部约束做条件生成或解逆问题。VFM 把思路从"引导采样路径"换成"学合适的初始噪声"：训练一个噪声适配器 $q_\phi(z|y)$，使噪声经 flow map 映射后既符合观测 $y$ 又符合数据先验，并用变分目标联合训练适配器与 flow map。`,
  motivation: R`扩散模型可以在每一步插入引导，而一步 flow map 没有这个机会（作者称之为 guidance gap）。`,
  method: R`
<p>对预训练 flow map $x=f_\theta(z)$ 与线性观测 $y=A(x)+\text{noise}$，噪声空间的后验为
$$p(z|y)\propto\exp\Big(-\frac{\|y-A(f_\theta(z))\|^2}{2\sigma^2}\Big)p(z).$$
用摊销变分推断训练 $q_\phi(z|y)\approx p(z|y)$（类似 VAE）。论文指出只训练适配器是不够的，于是提出联合训练适配器与 flow map 的变分目标，改善噪声–数据对齐。推理时 $z\sim q_\phi(\cdot|y)$，$x=f_\theta(z)$，仍是一步。</p>`,
  algo: [
    "y ~ p(y)（由数据经观测算子得到）",
    "z ~ q_φ(z | y)（重参数化）",
    "x = f_θ(z)",
    "loss = ‖y − A(x)‖²/(2σ²) + KL(q_φ(z|y) ‖ p(z)) + flow map 正则",
    "联合更新 φ, θ"
  ],
  theory: R`把条件生成重写为噪声空间的后验推断，给出原则性的变分目标。`,
  results: { head: ["任务", "说明"], rows: [["逆问题与条件生成", "一步完成，样本同时满足观测与数据先验"]], note: "" },
  limits: R`每类观测算子需要训练适配器；联合训练可能影响无条件生成质量。`,
  relations: R`和 iMF 把引导强度作为条件输入是两种不同的"把引导放进一步模型"的方案。`
},

fmgrpo: {
  src: "full",
  tldr: R`CM、MeanFlow 一类少步 flow map 生成器是确定性的，没有 RL 后训练所需的随机轨迹和可计算的似然比。Flow-Map GRPO 提出"锚定随机组合"（ASFMC）：确定性传输加上基于锚点的条件重采样，在保持边缘概率路径的前提下引入随机性，从而可以做在线 GRPO。`,
  motivation: R`Flow-GRPO 用等价 SDE 给速度型采样器加噪，但这只对小步长成立；flow map 的长程转移无法写成"确定部分 + 与输入无关的高斯扰动"。`,
  method: R`
<p>要求随机化后的映射 $\tilde\psi_{t\to r}$ 满足 $X_t\sim p_t\Rightarrow\tilde\psi_{t\to r}(X_t)\sim p_r$。ASFMC 先用确定性 flow map 把 $X_t$ 送到一个锚点（如局部锚点或终点锚点 $\hat x_1$），再从插值的条件分布中重采样得到 $X_r$。重采样是高斯的，所以每一步的转移概率可计算，可直接用于 GRPO 的似然比。论文给出了该构造保持边缘路径的条件，并设计了局部锚点和终点锚点两种可行策略。</p>`,
  algo: [
    "for 每个少步时间段 t → r:",
    "  a = ψ_θ(X_t → anchor)        # 确定性传输",
    "  X_r ~ N(α_r·a, σ_r²I) 型条件重采样   # 可算 log-prob",
    "按组计算奖励优势，GRPO 更新 θ（带 KL 正则）"
  ],
  theory: R`给出锚定随机组合保持边缘概率路径的条件。`,
  results: { head: ["实验", "说明"], rows: [["少步 flow map 生成器的 RL 后训练", "转移方式的消融说明保持边缘路径很重要"]], note: "" },
  limits: R`锚点选择影响探索；仍依赖奖励模型。`,
  relations: R`把 Flow-GRPO 扩展到少步 flow map；与 Meta Flow Maps 一样关注"一步模型如何做对齐"。`
}

});

Object.assign(window.PAPER_DETAILS, {

imf: {
  src: "full",
  tldr: R`指出原始 MeanFlow 的两个问题并修复：(1) 回归目标依赖网络自身，改写为由 $u_\theta$ 重参数化的瞬时速度 $v$ 损失，变成标准回归；(2) 训练时固定 CFG 强度，改为把引导作为显式条件输入，推理时可调。再用多 token 的 in-context 条件化处理多种条件。从头训练，ImageNet 256 上一步 FID 1.72。`,
  motivation: R`MeanFlow 的表观目标 $u_{\text{tgt}}$ 中含有网络本身，训练不像标准回归；固定的 CFG 强度不灵活，而且最优强度随模型大小、训练时长和步数变化。`,
  method: R`
<p><b>$v$-loss。</b>由 MeanFlow 恒等式定义复合预测
$$V_\theta\triangleq u_\theta+(t-r)\cdot\mathrm{JVP}_{\text{sg}},\qquad \mathcal L=\mathbb E\|V_\theta-v\|^2,$$
其中 $v=\epsilon-x$ 是固定目标，网络只出现在预测一侧。JVP 的切向量需要 $v_\theta$，可取 $u_\theta(z_t,t,t)$（边界条件）或一个辅助速度头。</p>
<p><b>灵活引导。</b>原始 MF 用固定的 $v_{\text{cfg}}=\omega v(z|c)+(1-\omega)v(z)$。iMF 把引导相关的量 $\Omega$（如引导强度）作为条件输入 $u_\theta(z_t\mid r,t,c,\Omega)$，训练时随机采样，推理时任意设置。</p>
<p><b>In-context 条件化。</b>adaLN-zero 把所有条件嵌入相加，条件多时负担过重。改为每种条件用多个可学习 token（类别 8 个，其它各 4 个），与图像 token 拼接后一起送入 Transformer；这样还减少了参数量。</p>`,
  algo: [
    "x, ε, (t, r), 采样引导条件 Ω;  z = (1−t)x + tε,  v = ε − x",
    "u = u_θ(z | r, t, c, Ω);  v_θ = 辅助头或 u_θ(z | t, t, c, Ω)",
    "JVP = d u/dt 沿 (v_θ, 0, 1)，stop-grad",
    "V = u + (t − r)·JVP",
    "loss = ‖V − v_target(Ω)‖²   # 目标中含按 Ω 构造的引导速度"
  ],
  theory: R`说明原始 MF 的 $u$-loss 可以等价地改写为由 $u_\theta$ 重参数化的 $v$-loss。`,
  results: { head: ["设置", "1-NFE FID"], rows: [
    ["iMF，ImageNet 256，从头训练", "1.72"],
    ["消融：原始 MF（无 / 有 CFG）", "32.69 / 6.17"],
    ["消融：v-loss，v_θ 取边界条件", "29.42 / 5.97"],
    ["消融：v-loss，v_θ 取辅助头", "30.76 / 5.68"]
  ], note: "CVPR 2026。作者指出一步生成越来越快之后，tokenizer（VAE 解码）本身的推理代价变得不可忽略，这也引出了后续的像素空间 pMF。" },
  limits: R`仍依赖 JVP 与潜空间 tokenizer。`,
  relations: R`MeanFlow → iMF → pMF 是一条清晰的主线；"MeanFlow 方差削减"一文把 iMF 的辅助速度头解释为控制变量。`
},

temf: {
  src: "full",
  tldr: R`MeanFlow 中 $r\neq t$ 的样本比例一大，性能就明显下降。原因是"时间尺度失衡"：不同时间间隔的样本对梯度的贡献差别很大。TEMF 用"时间均衡权重"均衡各尺度的梯度影响，再用"动态边界调度"让训练重心从稳定边界逐步转向精修整条轨迹。不改架构，ImageNet 256 上一步 FID 2.62、两步 2.30。`,
  motivation: R`两种需求相互冲突：长时间间隔的项会放大方差，而生成起点附近（$r\approx t$）又需要强约束。MeanFlow 固定的采样策略无法同时满足。`,
  method: R`
<p><b>时间尺度分析。</b>在温和的正则/曲率假设下，目标方差满足
$$\mathrm{Var}[u_{\text{tgt}}]\le\frac{C_1}{\Delta}+C_2\Delta^2+O(\Delta^3),\qquad \Delta=t-r,$$
而单样本梯度为 $\nabla_\theta\mathcal L=2(u_\theta-u_{\text{tgt}})\cdot\nabla_\theta u_\theta$，因此不同 $\Delta$ 的样本对梯度的贡献不均衡。</p>
<p><b>两个组件：</b>(1) 时间均衡加权函数，抵消短/长间隔之间天然的方差差异；(2) 动态边界调度，训练初期强调边界（$r\approx t$）的稳定，随后逐渐转向轨迹精修。</p>`,
  algo: [
    "按 MeanFlow 采样 (t, r)，Δ = t − r",
    "w(Δ) = 时间均衡权重",
    "边界样本比例按调度器随训练步数变化",
    "loss = w(Δ)·‖u_θ − sg(u_tgt)‖²"
  ],
  theory: R`给出 $u_{\text{tgt}}$ 方差关于时间间隔的上界，解释尺度失衡的来源。`,
  results: { head: ["设置", "NFE", "FID"], rows: [["ImageNet 256，带 CFG", "1", "2.62"], ["ImageNet 256", "2", "2.30"]], note: "论文图 2 显示 MeanFlow 的性能对 r≠t 的比例非常敏感。" },
  limits: R`权重与调度器有超参数；仍基于原始 MeanFlow 目标。`,
  relations: R`与"MeanFlow 方差削减"、ESC 都在处理 MeanFlow 目标的方差问题，角度分别是时间加权、控制变量和 plug-in 速度。`
}

});
window.PAPER_DETAILS.boffi_cm.src = "full";
window.PAPER_DETAILS.boffi_cm.results = { head: ["数据集", "发现"], rows: [
  ["棋盘格、CIFAR-10、CelebA-64、AFHQ-64", "LSD（Lagrangian，不需要空间导数，也不依赖从小步自举）在各基准上表现最好"],
  ["CelebA-64，LSD", "FID 从 1 步的 12.22 到 16 步的 1.96"],
  ["PSD（Progressive）", "与 LSD 的差距在 CelebA-64 低步数时更明显"]
], note: "作者按目标中空间导数、时间导数的数量来分析各方法表现。" };
