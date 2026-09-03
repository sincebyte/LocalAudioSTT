
# Table of Contents

1.  [WHY](#orgaf11bc0)
2.  [下载模型](#orga4e82f7)
3.  [编译](#org4142089)
4.  [启动](#org2378a21)
5.  [在 OpenChamber 里使用](#org3d98268)
6.  [在 debuff 里使用](#orgd726137)
7.  [性能测试](#orgdd5a798)



<a id="orgaf11bc0"></a>

# WHY

这是一个本地的语音转文字（ASR）服务。它把 FunASR 的 GGUF 模型跑在 llama.cpp 上，暴露一个 OpenAI 兼容的接口 \`POST /v1/audio/transcriptions\`。 也就是把本地录音文件转成文字。  

它的特点：  

-   ****不需要 GPU**** ：纯 CPU 推理，Mac 笔记本、Linux 小盒子都能跑
-   ****不需要 Python 运行时**** ：推理全部在 C++ 二进制里完成，只有一个很薄的  
    HTTP 包装层
-   ****数据不出本机**** ：录音文件全部在本地处理，不上传任何服务器
-   ****OpenAI 兼容接口**** ：OpenChamber、以及任何支持 OpenAI 语音接口的工具，  
    填个 URL 就能直接用


<a id="orga4e82f7"></a>

# 下载模型

本仓库默认跑的是  ****Fun-ASR-Nano**** （编码器 + Qwen3-0.6B 大语言模型 + FSMN-VAD 前端），模型都已经转换成 GGUF 格式，直接从 HuggingFace 下载 即可，不需要 Python 的深度学习环境。  

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">文件</th>
<th scope="col" class="org-left">说明</th>
<th scope="col" class="org-left">下载地址（HuggingFace）</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>funasr-encoder-f16.gguf</code></td>
<td class="org-left">SAN-M 音频编码器</td>
<td class="org-left"><a href="https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF">https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF</a></td>
</tr>

<tr>
<td class="org-left"><code>qwen3-0.6b-q5km.gguf</code></td>
<td class="org-left">Qwen3-0.6B 语言模型（5-bit 量化）</td>
<td class="org-left">同上，同一仓库</td>
</tr>

<tr>
<td class="org-left"><code>fsmn-vad.gguf</code></td>
<td class="org-left">语音活动检测（VAD，用于长音频切分）</td>
<td class="org-left"><a href="https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF">https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF</a></td>
</tr>
</tbody>
</table>

直接访问上面的 HuggingFace 仓库地址，把对应文件下载到 `gguf/` 目录即可。  


<a id="org4142089"></a>

# 编译

`llama-funasr-cli` 需要 ****&ndash;server 常驻模式**** 才能让服务只加载一次模型、常驻  
内存处理请求。\*\*官方预编译的二进制不带这个参数\*\* ，所以本方案不下载官方  
二进制，而是 ****直接从源码编译**** （编译脚本会自动打好下面的加速补丁）。  
编译产物不随本仓库 git 分发（已在 .gitignore 排除），换机器后重新编译一次即可。  

**需要安装的前置软件**  

编译前先确认下面几个软件已经装好：  

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">前置软件</th>
<th scope="col" class="org-left">作用</th>
<th scope="col" class="org-left">macOS 安装方式</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">git</td>
<td class="org-left">拉取 FunASR 源码</td>
<td class="org-left">随 Xcode 命令行工具自带；或 <code>brew install git</code></td>
</tr>

<tr>
<td class="org-left">cmake</td>
<td class="org-left">配置并编译（会自动拉取 llama.cpp 并构建）</td>
<td class="org-left"><code>brew install cmake</code></td>
</tr>

<tr>
<td class="org-left">C/C++ 工具链（clang/make）</td>
<td class="org-left">编译 C++ 源码</td>
<td class="org-left"><code>xcode-select --install</code></td>
</tr>
</tbody>
</table>

\`brew\`（Homebrew）还没装的话，先装：  
`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`  

装完后验证都能用：  

    git --version
    cmake --version
    clang --version

**补丁是做什么的**  

官方 CLI 是&ldquo;一次性&rdquo;版本：转完一个音频就退出，每个请求都要重新加载约 1GB 的模型。  
本仓库默认的 ****常驻模式**** （\`start-funasr-server.sh\` 拉起一个 \`llama-funasr-cli &ndash;server\` worker，模型只加载一次，处理音频更快）需要 `--server` 参数，官方二进制不带它，所以仓库放了补丁：  

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">文件</th>
<th scope="col" class="org-left">作用</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>patches/llama-funasr-cli.server-mode.patch</code></td>
<td class="org-left">给官方 CLI 源码新增 <code>--server</code> 常驻模式参数（可重复打，幂等）</td>
</tr>
</tbody>
</table>

**怎么编译并打好补丁**  

用仓库自带的一键脚本（自动完成：拉源码 → 打补丁 → 编译 →  得到 `llama-funasr-cli` ）：  

    ./build-funasr-cli-server.sh --install

验证补丁是否打上（编译成功的二进制在 &ndash;help 里应能看到 &ndash;server）：  

    ./llama-funasr-cli --help 2>&1 | grep -- --server


<a id="org2378a21"></a>

# 启动

确保以下文件存在：  

-   `llama-funasr-cli` ← 由上一步编译生成（不随 git 分发）
-   `gguf/qwen3-0.6b-q5km.gguf`
-   `gguf/funasr-encoder-f16.gguf`
-   `gguf/fsmn-vad.gguf`

缺执行文件的话，先按上节「编译 llama-funasr-cli」编译；缺模型的话，直接按  
上文地址从 HuggingFace 下载到 `gguf/` 目录。  

一键启动（默认端口 8001，会先杀掉占用 8001 的旧进程再启动）。脚本把服务 放到 ****后台**** 运行，打印一段日志后立即退出，并报告 ****启动成功/失败**** ：  

    ./start-funasr-server.sh

-   启动成功 → 输出 `启动成功 ✓  http://127.0.0.1:8001 (PID xxx)` ，脚本退出码 0
-   启动失败 → 输出 `启动失败 ✗` 并附日志尾部，脚本退出码 1
-   服务日志写在 `funasr-server/server.log`

验证是否启动成功：  

    curl http://127.0.0.1:8001/health          # 期望: {"status": "ok"}

也可以直接命令行测试转写：  

    curl http://127.0.0.1:8001/v1/audio/transcriptions \
      -F file=@/tmp/xxx.wav -F model=funasr-gguf

如需改端口/模型，用环境变量，例如 `FUNASR_PORT=9000 ./start-funasr-server.sh`  

停止服务：  

    kill <脚本输出的 PID>


<a id="org3d98268"></a>

# 在 OpenChamber 里使用

1.  先按上面步骤启动服务。
2.  打开 OpenChamber → 设置（Settings）→ 语音（Voice）→ 语音输入。
3.  提供者（Provider）选择 ****服务器 / Server**** （即 OpenAI 兼容）。
4.  填写以下三项：

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">字段</th>
<th scope="col" class="org-left">值</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">服务器 URL</td>
<td class="org-left"><code>http://127.0.0.1:8001/v1</code></td>
</tr>

<tr>
<td class="org-left">模型（Model）</td>
<td class="org-left">#ERROR</td>
</tr>

<tr>
<td class="org-left">API Key</td>
<td class="org-left">留空或随便填，如 <code>not-required</code></td>
</tr>
</tbody>
</table>

1.  语言（Language）留空即可（自动检测），或填 `zh` / `en`
2.  保存后，对着输入框说话，文字就会插入到对话里。

说明：OpenChamber 会把录音转成 WAV 并 POST 到  
`服务器 URL + /audio/transcriptions` ，所以填 `http://127.0.0.1:8001/v1` （ `127.0.0.1` 是允许的本地地址）。  


<a id="orgd726137"></a>

# 在 debuff 里使用

debuff 要求填 ****完整的转写接口地址**** （不能只填到 `/v1` ）。此时把下面 这个地址直接填成 URL：  

-   URL / 接口地址： `http://127.0.0.1:8001/v1/audio/transcriptions`
-   模型（Model）： `funasr-gguf` （任意字符串，服务端不校验）
-   API Key：留空或随便填，如 `not-required`


<a id="orgdd5a798"></a>

# 性能测试

测试机器：MacBook Pro 14 英寸，Apple M1 Pro（8 性能核 + 2 能效核）， 16GB 内存，macOS 26.6.2。  

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">项目</th>
<th scope="col" class="org-left">数值</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">服务常驻内存（Python 包装进程，空闲）</td>
<td class="org-left">约 13~23 MB</td>
</tr>

<tr>
<td class="org-left">单次转写时推理进程峰值内存</td>
<td class="org-left">约 1.56 GB</td>
</tr>

<tr>
<td class="org-left">模型磁盘占用</td>
<td class="org-left">编码器 469 MB + Qwen3 551 MB + VAD 1.7 MB ≈ 1.0 GB</td>
</tr>

<tr>
<td class="org-left">是否需要 GPU</td>
<td class="org-left">不需要</td>
</tr>

<tr>
<td class="org-left">转写速度（10 秒音频，常驻模式，模型已加载）</td>
<td class="org-left">约 0.5 秒</td>
</tr>

<tr>
<td class="org-left">转写速度（20 秒音频，常驻模式，模型已加载）</td>
<td class="org-left">约 1.0 秒</td>
</tr>

<tr>
<td class="org-left">转写速度（50 秒音频，常驻模式，模型已加载）</td>
<td class="org-left">约 3.5 秒</td>
</tr>
</tbody>
</table>

服务默认以 ****常驻模式**** 运行：启动时加载一次模型（约 1~1.5 秒），之后每个请求 直接复用内存里的模型，不再重新加载。实测耗时：  

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">音频时长</th>
<th scope="col" class="org-left">常驻模式</th>
<th scope="col" class="org-left">每请求拉起子进程（旧模式）</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">10 秒</td>
<td class="org-left">~0.5 秒</td>
<td class="org-left">~1.4~2.5 秒</td>
</tr>

<tr>
<td class="org-left">20 秒</td>
<td class="org-left">~1.0 秒</td>
<td class="org-left">~1.8~2.5 秒</td>
</tr>

<tr>
<td class="org-left">30 秒</td>
<td class="org-left">~1.5 秒</td>
<td class="org-left">~2.5~2.9 秒</td>
</tr>

<tr>
<td class="org-left">40 秒</td>
<td class="org-left">~3.1 秒</td>
<td class="org-left">~4.0~4.4 秒</td>
</tr>

<tr>
<td class="org-left">50 秒</td>
<td class="org-left">~3.5 秒</td>
<td class="org-left">~4.5~5.0 秒</td>
</tr>
</tbody>
</table>

