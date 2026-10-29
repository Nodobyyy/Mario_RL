import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDownToLine, ArrowRight, Check, ChevronDown, ChevronRight, CirclePlay, Cpu, Flag, Gamepad2, GitBranch, History, LayoutDashboard, Maximize2, Pause, Play, Plus, Radio, RotateCcw, Settings2, Square, Terminal, X, Zap } from 'lucide-react';
import * as echarts from 'echarts';
import './style.css';
import './extra.css';

type Job = { id:string; kind:string; name:string; status:string; created_at:string; config:Record<string,any>; state:Record<string,any>; error?:string };
type Checkpoint = {id:string;name:string;steps:number;best:number;job_id?:string;algorithm?:string;world?:number;stage?:number};
type Episode = {id:number; steps:number; reward:number;max_x:number;final_x:number;success:number;death:number;reason:string;worker_id:number;kind:string};
type Point = {step:number;x_pos:number;y_pos:number;action:number;reward:number;done:boolean};
const statusNames:Record<string,string> = {starting:'正在初始化',running:'运行中',pausing:'等待安全暂停',paused:'已暂停',stopping:'正在安全停止',stopped:'已停止',completed:'已完成',failed:'运行异常'};
const busy = (job?:Job) => !!job && ['starting','running','pausing','paused','stopping'].includes(job.status);
const num = (v:any, digits=0) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '—';
const duration = (v:any) => typeof v === 'number' ? `${String(Math.floor(v/3600)).padStart(2,'0')}:${String(Math.floor(v/60)%60).padStart(2,'0')}:${String(Math.floor(v)%60).padStart(2,'0')}` : '—';
const date = (v:string) => new Date(v).toLocaleString('zh-CN',{hour12:false});
async function api<T=any>(url:string, body?:any):Promise<T> {
  const res = await fetch('/api'+url, body === undefined ? undefined : {method:'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await res.json();
  if(!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data));
  return data;
}

function useFrame(id?:string, kind='runs') {
  const [frame,setFrame] = useState<any>(null);
  const [online,setOnline] = useState(false);
  const [displayFps,setDisplayFps] = useState(0);
  useEffect(()=>{
    setFrame(null); setOnline(false); setDisplayFps(0);
    if(!id) return;
    let ws:WebSocket, timer:ReturnType<typeof setTimeout>, closed=false;
    let latestFrame:any, paintRequest=0, painted=0, fpsStarted=performance.now();
    const paint=()=>{
      paintRequest=0;
      setFrame(latestFrame);
      painted++;
      const elapsed=performance.now()-fpsStarted;
      if(elapsed>=1000){setDisplayFps(painted*1000/elapsed);painted=0;fpsStarted=performance.now()}
    };
    const connect=()=>{
      ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/${kind}/${id}/frames`);
      ws.onopen=()=>setOnline(true);
      ws.onmessage=e=>{try {const value=JSON.parse(e.data);if(value.type==='frame'){latestFrame=value;if(!paintRequest)paintRequest=requestAnimationFrame(paint)}}catch{}};
      ws.onerror=()=>ws.close();
      ws.onclose=()=>{setOnline(false);if(!closed) timer=setTimeout(connect,2000)};
    };
    connect();
    return ()=>{closed=true;clearTimeout(timer);cancelAnimationFrame(paintRequest);ws?.close()};
  },[id,kind]);
  return {frame,online,displayFps};
}

function Chart({option, height=245}:{option:any;height?:number}) {
  const ref=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    if(!ref.current) return;
    const chart=echarts.init(ref.current,undefined,{renderer:'canvas'});
    chart.setOption({backgroundColor:'transparent',textStyle:{fontFamily:'Inter, Segoe UI, Microsoft YaHei, sans-serif'},...option});
    const observer=new ResizeObserver(()=>chart.resize());observer.observe(ref.current);
    return ()=>{observer.disconnect();chart.dispose()};
  },[option]);
  return <div ref={ref} style={{height,width:'100%'}}/>;
}
const axis = {axisLine:{lineStyle:{color:'#303744'}},axisLabel:{color:'#727f91',fontSize:10},splitLine:{lineStyle:{color:'#222b36',type:'dashed'}}};
function LineChart({data, keys, colors=['#b5f375','#6bbdff'],labels}:{data:any[];keys:string[];colors?:string[];labels?:string[]}) {
  const valid=data.some(p=>keys.some(k=>typeof p[k]==='number'));
  if(!valid) return <Empty small title="等待真实训练数据" detail="完成采样或梯度更新后，曲线将在这里出现。"/>;
  return <Chart option={{tooltip:{trigger:'axis'},grid:{left:48,right:18,top:28,bottom:34},
    legend:keys.length>1?{textStyle:{color:'#8b98aa'},top:0}:undefined,
    xAxis:{...axis,type:'value',name:'环境步',nameLocation:'middle',nameGap:26,nameTextStyle:{color:'#627084',fontSize:10}},
    yAxis:{...axis,type:'value',scale:true},
    series:keys.map((k,i)=>({name:labels?.[i]??k,type:'line',showSymbol:false,connectNulls:false,sampling:'lttb',
      data:data.map(p=>[p.num_timesteps,p[k]??null]),lineStyle:{width:2,color:colors[i%colors.length]},itemStyle:{color:colors[i%colors.length]},areaStyle:keys.length===1?{color:colors[0],opacity:.04}:undefined}))}}/>;
}

function StageSelect({world,stage,onChange}:{world:number;stage:number;onChange:(world:number,stage:number)=>void}) {
  return <label>游戏关卡<select aria-label="游戏关卡" value={`${world}-${stage}`} onChange={e=>{const [w,s]=e.target.value.split('-').map(Number);onChange(w,s)}}>{Array.from({length:8},(_,w)=>Array.from({length:4},(_,s)=><option key={`${w+1}-${s+1}`} value={`${w+1}-${s+1}`}>世界 {w+1} · 第 {s+1} 关</option>))}</select></label>;
}
function Empty({title,detail,small=false,children}:{title:string;detail?:string;small?:boolean;children?:React.ReactNode}) {
  return <div className={'empty '+(small?'compact':'')}><div className="empty-icon"><Gamepad2 size={small?22:34}/></div><strong>{title}</strong><p>{detail}</p>{children}</div>;
}
function Badge({status}:{status:string}) {return <span className={'badge '+status}><i/>{statusNames[status]??status}</span>}
function PanelHead({title,subtitle,right}:{title:string;subtitle?:string;right?:React.ReactNode}) {return <div className="panel-head"><div><h3>{title}</h3>{subtitle&&<p>{subtitle}</p>}</div>{right}</div>}

function GameView({job,frame,online,displayFps=0,evaluation=false,onPreviewChange,previewPending=false}:{job?:Job;frame:any;online:boolean;displayFps?:number;evaluation?:boolean;onPreviewChange?:(fps:number)=>void;previewPending?:boolean}) {
  const [expanded,setExpanded]=useState(false);
  const stale=frame&&Date.now()-new Date(frame.timestamp).getTime()>5000;
  return <div className={'panel game-panel '+(expanded?'expanded':'')}>
    <PanelHead title={evaluation?'模型评估画面':'实时环境预览'} subtitle={evaluation?'独立 NES 环境 · 策略推理':'训练 worker 0 · 真实采样画面'} right={<div className="inline frame-controls">{!evaluation&&<select className="frame-rate-control" aria-label="预览目标帧率" title="调整预览目标帧率，不改变训练速度" value={job?.config.preview_fps??30} disabled={!onPreviewChange||!job||!['starting','running','pausing','paused'].includes(job.status)||previewPending} onChange={e=>onPreviewChange?.(Number(e.target.value))}>{Array.from(new Set([0,8,15,30,60,job?.config.preview_fps??30])).sort((a,b)=>a-b).map(fps=><option key={fps} value={fps}>{fps?`${fps} FPS`:'关闭预览'}</option>)}</select>}<span className="micro-label"><i className={online?'dot green':'dot'}/>{online?'已连接':'未连接'}</span><button className="icon-button" title="放大画面" onClick={()=>setExpanded(!expanded)}>{expanded?<X size={16}/>:<Maximize2 size={16}/>}</button></div>}/>
    <div className="game-screen">
      <div className="screen-label"><span className={'mode '+(evaluation?'blue':'')}>{evaluation?'独立评估':'训练预览'}</span><span>SUPER MARIO BROS. / {job?.state.world??job?.config.world??1}–{job?.state.stage??job?.config.stage??1}</span></div>
      {frame?<img className="game-image" src={frame.image} alt={evaluation?'独立评估的真实 Mario 游戏画面':'训练 worker 0 的真实 Mario 游戏画面'}/>:<Empty title={busy(job)?'正在初始化 Mario 环境':'环境已就绪，等待启动'} detail={busy(job)?'加载模型和 NES 模拟器后显示真实游戏帧。':'开始训练或切换到模型演示，观察 Agent 的每一步。'}/>}
      <div className="screen-bottom"><span><Radio size={12}/>{!frame?'NO SIGNAL':stale||!busy(job)||job?.status==='paused'?'最近采样帧':`LIVE · ${num(displayFps,1)} FPS`}</span><span>256 × 240 · NES RGB</span></div>
    </div>
    <div className="game-caption"><span><i className="dot green"/>{evaluation?'使用选定 checkpoint 的独立评估，不计入训练步数':'画面来自训练环境，传输拥塞时自动丢弃过期帧'}</span><span>{frame?new Date(frame.timestamp).toLocaleTimeString('zh-CN'): '等待第一帧'}</span></div>
  </div>;
}

function App(){
  const [page,setPage]=useState('overview');
  const [system,setSystem]=useState<any>(null),[connected,setConnected]=useState(false);
  const [runs,setRuns]=useState<Job[]>([]),[checkpoints,setCheckpoints]=useState<Checkpoint[]>([]),[evals,setEvals]=useState<Job[]>([]);
  const [selected,setSelected]=useState(localStorage.getItem('mario-run')??''),[metrics,setMetrics]=useState<any[]>([]);
  const [selectedEval,setSelectedEval]=useState('');
  const [showNew,setShowNew]=useState(false),[error,setError]=useState(''),[pending,setPending]=useState(false);
  const [logs,setLogs]=useState<string|null>(null);
  const job=runs.find(r=>r.id===selected)??runs[0];
  const refresh=async()=>{
    try {const [hw,r,cp,e]=await Promise.all([api('/system/status'),api<Job[]>('/runs'),api<Checkpoint[]>('/checkpoints'),api<Job[]>('/evaluations')]);setSystem(hw);setRuns(r);setCheckpoints(cp);setEvals(e);setConnected(true)}
    catch{setConnected(false)}
  };
  useEffect(()=>{refresh();const t=setInterval(refresh,2000);return()=>clearInterval(t)},[]);
  useEffect(()=>{if(job) localStorage.setItem('mario-run',job.id)},[job?.id]);
  useEffect(()=>{
    setMetrics([]);if(!job) return;
    let disposed=false;
    const pull=()=>api<any[]>(`/runs/${job.id}/metrics`).then(x=>{if(!disposed)setMetrics(x)}).catch(()=>{});
    pull();const t=setInterval(pull,3000);return()=>{disposed=true;clearInterval(t)};
  },[job?.id]);
  // Metrics WS restores the snapshot on reconnect; durable curves are fetched separately.
  useEffect(()=>{
    if(!job) return;let ws:WebSocket,t:ReturnType<typeof setTimeout>,closed=false;
    const connect=()=>{ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/runs/${job.id}/metrics`);
      ws.onmessage=e=>{try{const m=JSON.parse(e.data);setRuns(rows=>rows.map(r=>r.id===m.job_id?{...r,status:m.status,state:m.state,error:m.error}:r))}catch{}};
      ws.onerror=()=>ws.close();ws.onclose=()=>{if(!closed)t=setTimeout(connect,2000)};};
    connect();return()=>{closed=true;clearTimeout(t);ws?.close()};
  },[job?.id]);
  const act=async(fn:()=>Promise<any>)=>{setPending(true);setError('');try{await fn();await refresh()}catch(e){setError((e as Error).message)}finally{setPending(false)}};
  const command=(action:string)=>job&&act(()=>api(`/runs/${job.id}/${action}`,{}));
  const nav=[['overview',LayoutDashboard,'训练总览'],['trajectories',GitBranch,'轨迹分析'],['history',History,'训练历史'],['demo',CirclePlay,'模型演示']] as const;
  const state=job?.state??{};
  const deviceJob=page==='demo'?(evals.find(e=>e.id===selectedEval)??evals.find(busy)??evals[0]):job;
  const actualDevice=deviceJob?.state.device;
  const deviceLabel=deviceJob?(actualDevice?.startsWith('cuda')?'CUDA GPU 计算':actualDevice==='cpu'?'CPU 计算模式':'设备初始化中'):(system?.cuda?'CUDA 可用 · 等待任务':'CPU 可用 · 等待任务');
  const deviceDetail=deviceJob?.state.device_fallback?'CUDA 不可用，已回退 CPU':actualDevice?.startsWith('cuda')?(system?.gpu??actualDevice):`PyTorch ${system?.torch??'…'} · ${deviceJob?.config.threads??system?.defaults.threads??'—'} 线程`;
  const progress=state.target_timesteps?Math.min(100,100*(state.num_timesteps-state.start_timesteps)/(state.target_timesteps-state.start_timesteps)):0;
  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#" onClick={e=>{e.preventDefault();setPage('overview')}}><div className="brand-mark">M<span>•</span></div><div>MARIO<span>REINFORCEMENT LAB</span></div></a>
      <div className="workspace-label">WORKSPACE <span>LOCAL</span></div><nav>{nav.map(([key,Icon,label])=><button key={key} className={page===key?'nav active':'nav'} onClick={()=>setPage(key)}><Icon size={19}/>{label}{page===key&&<span className="nav-marker"/>}</button>)}</nav>
      <div className="sidebar-session"><div className="tiny-title">当前实验</div><strong>{job?.name??'尚未创建实验'}</strong><span>{job?`#${job.id.slice(0,8)}`:'创建你的第一个 PPO 任务'}</span>{job&&<Badge status={job.status}/>}</div>
      <div className="sidebar-bottom"><div className="hardware-icon"><Cpu size={19}/></div><div><strong>{deviceLabel}</strong><span>{deviceDetail}</span></div><i className={connected?'dot green':'dot'}/></div>
    </aside>
    <main><header className="topbar"><div className="breadcrumb">工作空间 <ChevronRight size={13}/> <span>{nav.find(n=>n[0]===page)?.[2]}</span></div><div className="inline"><span className="top-chip"><GitBranch size={13}/>{(deviceJob?.state.algorithm??deviceJob?.config.algorithm??'ppo').toUpperCase()} BASELINE</span><span className={'connection '+(connected?'ok':'')}><i/>{connected?'本地服务在线':'正在连接服务'}</span></div></header>
      <div className="content"><div className="page-heading"><div><div className="eyebrow">MARIO RL / WORLD {deviceJob?.state.world??deviceJob?.config.world??1}–{deviceJob?.state.stage??deviceJob?.config.stage??1}</div><h1>{nav.find(n=>n[0]===page)?.[2]}<span className="heading-dot">.</span></h1><p>{page==='overview'?'从每一次尝试中学习，直到越过终点。':page==='trajectories'?'沿着每一步，理解 Agent 如何探索关卡。':page==='history'?'保存实验，让每一次进步都有迹可循。':'加载训练策略，在独立环境中检验学习成果。'}</p></div><div className="heading-actions">{runs.length>0&&page!=='demo'&&<div className="select-wrap"><select aria-label="选择训练任务" value={job?.id??''} onChange={e=>setSelected(e.target.value)}>{runs.map(r=><option key={r.id} value={r.id}>{r.name} · {r.id.slice(0,6)}</option>)}</select><ChevronDown size={14}/></div>}<button className="button primary" onClick={()=>setShowNew(true)}><Settings2 size={16}/>算法与关卡设置 / 新建训练</button></div></div>
      {!connected&&<div className="notice"><Terminal size={16}/>本地 API 尚未连接。请运行项目根目录的“启动.bat”；Vite 开发模式需要同时启动后端。</div>}
      {system?.missing?.length>0&&<div className="error-box">缺少运行依赖：{system.missing.join(', ')}。请运行安装脚本。</div>}
      {error&&<div role="alert" className="error-box">{error}<button className="icon-button" onClick={()=>setError('')}><X size={15}/></button></div>}
      {page==='overview'&&<>
        <div className="stats-grid">{[
          {label:'累计环境步数',value:num(state.num_timesteps),unit:'STEPS',icon:Zap,foot:state.target_timesteps?`目标 ${num(state.target_timesteps)} · 本次新增 ${num(job?.config.total_timesteps)}`:'SB3 timesteps · 非原始 NES 帧'},
          {label:'已完成回合',value:num(state.episodes),unit:'EPISODES',icon:Gamepad2,foot:'仅统计训练回合 · 评估单独记录'},
          {label:'最远到达位置',value:num(state.max_x),unit:'px',icon:Flag,foot:'世界 X 坐标 · 通关由 flag_get 判定'},
          {label:'近期评估通关率',value:state.eval_success_rate!=null?num(state.eval_success_rate*100,1):'—',unit:'%',icon:Activity,foot:state.evaluation_id?`独立评估 #${state.evaluation_id.slice(0,8)}`:'尚未完成独立评估'}
        ].map(({label,value,unit,icon:Icon,foot})=><div className="stat-card" key={label}><div className="stat-label">{label}<Icon size={16}/></div><div className="stat-value">{value}<small>{unit}</small></div><div className="stat-foot">{foot}</div></div>)}</div>
        <div className="run-strip"><div className="inline">{job?<Badge status={job.status}/>:<span className="badge">等待任务</span>}<span className="run-name">{job?.name??'开始你的第一次强化学习实验'}</span><span className="muted mono">{job?`#${job.id.slice(0,8)}`:''}</span></div><div className="run-controls"><span className="muted">{duration(state.elapsed_seconds)}</span><span className="fps">{num(state.fps,1)} <small>FPS</small></span><div className="separator"/>{job&&<><button className="button subtle" disabled={pending||!['running','paused','pausing'].includes(job.status)} onClick={()=>command(job.status==='running'?'pause':'resume')}>{job.status==='running'?<Pause size={14}/>:<Play size={14}/>} {job.status==='running'?'暂停':'继续'}</button><button className="button danger-outline" disabled={pending||!busy(job)||job.status==='stopping'} onClick={()=>command('stop')}><Square size={12}/>安全停止</button></>}</div><div className="run-progress" style={{width:`${progress}%`}}/></div>
        {job?.error&&<div className="error-box">{job.error}<button className="text-button" onClick={()=>act(async()=>setLogs((await api(`/jobs/${job.id}/log`)).text))}>查看日志</button></div>}
        <TrainingPreview job={job} progress={progress} pending={pending} onPreviewChange={fps=>job&&act(()=>api(`/runs/${job.id}/preview`,{fps}))} showLogs={()=>job&&act(async()=>setLogs((await api(`/jobs/${job.id}/log`)).text))}/>
        <div className="section-title"><h2>学习动态 <span>LEARNING CURVES</span></h2>{job&&<a className="text-button" href={`/api/runs/${job.id}/metrics/export`}><ArrowDownToLine size={14}/>导出原始指标</a>}</div>
        <div className="charts-grid"><div className="panel"><PanelHead title="平均回合奖励" subtitle="最近 100 个训练回合 · 未归一化奖励" right={<span className="chart-value lime">{num(state.mean_reward,1)}</span>}/><LineChart data={metrics} keys={['mean_reward']}/></div><div className="panel"><PanelHead title="探索进度" subtitle="关卡最远世界 X 位置" right={<span className="chart-value blue-text">{num(state.max_x)} <small>px</small></span>}/><LineChart data={metrics} keys={['max_x']} colors={['#6bbdff']}/></div><div className="panel"><PanelHead title="回合长度" subtitle="最近 100 个训练回合 · 环境步"/><LineChart data={metrics} keys={['mean_length']} colors={['#f6bf75']}/></div></div>
        <div className="charts-grid secondary-charts"><div className="panel"><PanelHead title="独立评估通关率" subtitle="0–1 · 仅使用 flag_get 成功信号"/><LineChart data={metrics} keys={['eval_success_rate']} colors={['#8fd3c6']}/></div><div className="panel"><PanelHead title="策略 / 价值损失" subtitle="来自 Stable-Baselines3 梯度更新日志"/><LineChart data={metrics} keys={['policy_gradient_loss','value_loss']} labels={['Policy','Value']}/></div><div className="panel"><PanelHead title="策略熵" subtitle="Policy entropy · 衡量动作探索程度"/><LineChart data={metrics} keys={['policy_entropy']} colors={['#bb9df6']}/></div></div>
      </>}
      {page==='trajectories'&&<Trajectories job={job} evaluations={evals} onError={setError}/>}
      {page==='history'&&<HistoryPage runs={runs} select={(id)=>{setSelected(id);setPage('overview')}}/>}
      {page==='demo'&&<Demo selected={selectedEval} setSelected={setSelectedEval} checkpoints={checkpoints} evaluations={evals} act={act} pending={pending}/>}
      <footer><span><i className="dot green"/>REAL ENVIRONMENTS. REAL LEARNING.</span><span>Mario Lab <b>/</b> PPO + A2C <b>/</b> 所有指标来自真实环境</span></footer>
      </div>
    </main>
    {showNew&&<NewRun system={system} checkpoints={checkpoints} close={()=>setShowNew(false)} create={config=>act(async()=>{const r=await api<Job>('/runs',config);setSelected(r.id);setPage('overview');setShowNew(false)})} pending={pending}/>}
    {logs!==null&&<div className="modal-backdrop"><div className="modal wide"><div className="modal-title"><div><h2>进程日志</h2><p>原始训练 / 运行输出</p></div><button className="icon-button" onClick={()=>setLogs(null)}><X/></button></div><pre className="log-output">{logs||'等待进程输出…'}</pre></div></div>}
  </div>;
}

function TrainingPreview({job,progress,pending,onPreviewChange,showLogs}:{job?:Job;progress:number;pending:boolean;onPreviewChange:(fps:number)=>void;showLogs:()=>void}) {
  // Keep high-frequency frames local: learning curves do not rerender per frame.
  const {frame,online,displayFps}=useFrame(job?.id);
  const state=job?.state??{};
  return <div className="overview-middle"><GameView job={job} frame={frame} online={online} displayFps={displayFps} onPreviewChange={onPreviewChange} previewPending={pending}/><div className="panel agent-panel"><PanelHead title="Agent 实时状态" right={<span className="tiny-tag">WORKER 0</span>}/><AgentState current={frame?.state??state.current} state={state} job={job}/><div className="training-progress"><div><span>本次训练进度</span><b>{num(progress,1)}%</b></div><div className="progress-track"><i style={{width:`${progress}%`}}/></div><small>完整 rollout 会使实际步数略超目标</small></div><button className="inspect-button" disabled={!job} onClick={showLogs}><Terminal size={15}/>查看进程日志<ArrowRight size={14}/></button></div></div>;
}

function AgentState({current,state,job}:{current:any;state:any;job?:Job}) {
 const actions=['NOOP','→ RIGHT','→ + A 跳跃','→ + B 奔跑','→ + A + B','A 原地跳跃','← LEFT'];
 return <div className="agent-state"><div className="action-display"><span>当前动作 / ACTION</span><strong>{current?.action!=null?actions[current.action]??current.action:'等待采样'}<Gamepad2 size={22}/></strong></div><dl>{[
   ['Mario 世界坐标',current?`X ${num(current.x_pos)} / Y ${num(current.y_pos)}`:'—'],['当前回合奖励',num(current?.episode_reward,2)],['当前回合步数',num(current?.step)],['学习率',state.learning_rate?.toExponential(2)??'—'],['计算设备',state.device??'等待设备确认'],['GPU 张量显存',state.gpu_memory_mb!=null?`${num(state.gpu_memory_mb)} MB`:'—']
 ].map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></div>;
}

function NewRun({system,checkpoints,close,create,pending}:{system:any;checkpoints:Checkpoint[];close:()=>void;create:(config:any)=>void;pending:boolean}) {
  const [cfg,setCfg]=useState<any>({name:'Mario 实验',algorithm:'ppo',world:1,stage:1,total_timesteps:20000,device:'auto',n_envs:1,n_steps:128,batch_size:64,n_epochs:4,threads:4,learning_rate:.00025,gamma:.9,ent_coef:.05,checkpoint_freq:5000,eval_freq:25000,eval_episodes:3,eval_steps:4000,preview_fps:30,seed:42,vec_backend:'dummy',resume_checkpoint:null,...system?.defaults});
  const [advanced,setAdvanced]=useState(false);
  const field=(key:string,label:string,type='number',step?:string)=><label>{label}<input type={type} required value={cfg[key]} step={step} onChange={e=>setCfg({...cfg,[key]:type==='number'?Number(e.target.value):e.target.value})}/></label>;
  return <div className="modal-backdrop"><form className="modal" onSubmit={e=>{e.preventDefault();create(cfg)}}><div className="modal-title"><div><div className="eyebrow">NEW EXPERIMENT</div><h2>创建训练任务</h2><p>PPO / A2C 训练 · Super Mario Bros.</p></div><button type="button" className="icon-button" onClick={close}><X/></button></div>
    <div className="profile-note"><Cpu size={18}/><div><strong>{system?.cuda?'检测到 CUDA，已应用 GPU 配置':'已应用 CPU 兼容配置'}</strong><span>相同策略结构与预处理，可跨 CPU / CUDA 加载权重。</span></div></div>
    <div className="form-grid"><label>训练算法<select aria-label="训练算法" value={cfg.algorithm} onChange={e=>setCfg({...cfg,algorithm:e.target.value,resume_checkpoint:null})}><option value="ppo">PPO 近端策略优化</option><option value="a2c">A2C 优势演员评论家</option></select></label><StageSelect world={cfg.world} stage={cfg.stage} onChange={(world,stage)=>setCfg({...cfg,world,stage})}/></div><p className="field-note">设置作用于新任务。切换算法会清空起始权重；跨关卡继续训练将使用所选关卡，通关能力需要重新验证。</p>
    {field('name','任务名称','text')}<div className="form-grid">{field('total_timesteps','新增训练环境步数')}<label>设备<select value={cfg.device} onChange={e=>setCfg({...cfg,device:e.target.value})}><option value="auto">自动检测</option><option value="cpu">纯 CPU</option><option value="cuda">CUDA（不可用时回退 CPU）</option></select></label><label>并行环境<select value={cfg.n_envs} onChange={e=>setCfg({...cfg,n_envs:Number(e.target.value)})}>{[1,4,8].map(n=><option key={n} value={n}>{n} 个环境</option>)}</select></label>{field('threads','PyTorch CPU 线程数')}</div>
    <label>起始权重<select value={cfg.resume_checkpoint??''} onChange={e=>setCfg({...cfg,resume_checkpoint:e.target.value||null})}><option value="">从零开始训练</option>{checkpoints.filter(cp=>(cp.algorithm??'ppo')===cfg.algorithm).map(cp=><option value={cp.id} key={cp.id}>{cp.name} · {(cp.algorithm??'ppo').toUpperCase()} · {cp.world??1}-{cp.stage??1}</option>)}</select></label><p className="field-note">恢复会保留累计 timesteps、策略、优化器和匹配的奖励归一化统计；不恢复模拟器中途状态。</p>
    <button type="button" className="advanced-toggle" onClick={()=>setAdvanced(!advanced)}><Settings2 size={15}/>算法与采集参数<ChevronDown size={14}/></button>
    {advanced&&<div className="form-grid advanced-fields">{field('n_steps','Rollout 长度')}{cfg.algorithm==='ppo'&&<>{field('batch_size','Batch size')}{field('n_epochs','更新轮数')}</>}{field('learning_rate','学习率','number','0.00001')}{field('gamma','Gamma','number','0.01')}{field('ent_coef','熵系数','number','0.01')}{field('checkpoint_freq','Checkpoint 间隔')}{field('eval_freq','独立评估间隔（0 关闭）')}{field('eval_episodes','每次评估回合数')}{field('eval_steps','评估单局步数上限')}{field('preview_fps','预览 FPS（0–60，0 关闭）')}{field('seed','随机种子')}<label>向量环境后端<select value={cfg.vec_backend} onChange={e=>setCfg({...cfg,vec_backend:e.target.value})}><option value="dummy">DummyVecEnv（低内存）</option><option value="subproc">SubprocVecEnv（独立 worker）</option></select></label></div>}
    <div className="modal-footer"><span>CPU 从零学习通关需要长时间训练。</span><button className="button primary" disabled={pending} type="submit"><Play size={15}/>{pending?'正在创建…':'启动真实训练'}</button></div>
  </form></div>;
}

function HistoryPage({runs,select}:{runs:Job[];select:(id:string)=>void}) {
  const [details,setDetails]=useState<Job|null>(null);
  if(!runs.length) return <div className="panel"><Empty title="实验档案从这里开始" detail="创建训练任务后，配置、成绩、曲线与 checkpoint 都会保存在本地。"/></div>;
  return <><div className="panel"><PanelHead title="全部训练任务" subtitle={`${runs.length} 个本地实验 · SQLite 持久化保存`}/><div className="table-scroll"><table><thead><tr><th>实验名称 / ID</th><th>创建时间</th><th>算法 / 设备</th><th>累计步数</th><th>回合 / 最远位置</th><th>状态</th><th>操作</th></tr></thead><tbody>{runs.map(r=><tr key={r.id}><td><strong>{r.name}</strong><small className="mono">#{r.id.slice(0,8)}</small></td><td>{date(r.created_at)}</td><td>{(r.config.algorithm??'ppo').toUpperCase()} · {r.config.world??1}-{r.config.stage??1} <small>{r.state.device??r.config.device} · {r.config.n_envs} envs</small></td><td className="mono">{num(r.state.num_timesteps)}</td><td>{num(r.state.episodes)} / {num(r.state.max_x)} px</td><td><Badge status={r.status}/></td><td><div className="inline"><button className="text-button" onClick={()=>select(r.id)}>查看曲线<ArrowRight size={13}/></button><button className="icon-button" title="配置详情" onClick={()=>setDetails(r)}><Settings2 size={15}/></button></div></td></tr>)}</tbody></table></div></div>{details&&<div className="modal-backdrop"><div className="modal"><div className="modal-title"><h2>{details.name}</h2><button className="icon-button" onClick={()=>setDetails(null)}><X/></button></div><pre className="log-output">{JSON.stringify({config:details.config,best_checkpoint:details.state.best_checkpoint,latest_checkpoint:details.state.latest_checkpoint,error:details.error},null,2)}</pre></div></div>}</>;
}

function Trajectories({job:trainingJob,evaluations,onError}:{job?:Job;evaluations:Job[];onError:(s:string)=>void}) {
  const [source,setSource]=useState('');
  const job=evaluations.find(e=>e.id===source)??trainingJob;
  const [episodes,setEpisodes]=useState<Episode[]>([]),[total,setTotal]=useState(0),[offset,setOffset]=useState(0),[filter,setFilter]=useState('all'),[search,setSearch]=useState('');
  const [chosen,setChosen]=useState<number[]>([]),[traces,setTraces]=useState<{episode:Episode;points:Point[]}[]>([]),[step,setStep]=useState(0);
  useEffect(()=>{setChosen([]);setTraces([]);setOffset(0);setSource('')},[trainingJob?.id]);
  useEffect(()=>{
    if(!job){setEpisodes([]);return}let closed=false;
    const pull=()=>{const params=new URLSearchParams({limit:'50',offset:String(offset)});if(filter!=='all')params.set('success',String(filter==='success'));if(search&&/^\d+$/.test(search))params.set('episode_id',search);
      api(`/runs/${job.id}/episodes?${params}`).then(x=>{if(!closed){setEpisodes(x.items);setTotal(x.total)}}).catch(e=>onError(e.message))};
    pull();const t=setInterval(pull,4000);return()=>{closed=true;clearInterval(t)};
  },[job?.id,filter,search,offset]);
  useEffect(()=>{let closed=false;Promise.all(chosen.map(async id=>{const [episode,traj]=await Promise.all([api<Episode>(`/episodes/${id}`),api(`/episodes/${id}/trajectory`)]);return {episode,points:traj.points as Point[]}})).then(x=>{if(!closed){setTraces(x);setStep(0)}}).catch(e=>onError(e.message));return()=>{closed=true}},[chosen.join(',')]);
  const current=traces[0]?.points[step], colors=['#b5f375','#6bbdff','#f6bf75','#bb9df6'];
  const toggle=(id:number)=>setChosen(prev=>prev.includes(id)?prev.filter(x=>x!==id):[...prev.slice(-3),id]);
  return <div className="trajectory-layout"><div className="panel episode-list"><PanelHead title="Episode 档案" subtitle={`${total} 条记录 · 最多对比 4 条轨迹`}/><div className="episode-tools"><label className="source-label">记录来源（跨来源保留已选路径）<select aria-label="轨迹记录来源" value={source} onChange={e=>{setSource(e.target.value);setOffset(0)}}><option value="">当前训练任务</option>{evaluations.map(e=><option key={e.id} value={e.id}>评估 #{e.id.slice(0,8)} · {e.state.flags_cleared??0} 次通关</option>)}</select></label><input placeholder="搜索 Episode ID" aria-label="搜索 Episode ID" value={search} onChange={e=>{setSearch(e.target.value);setOffset(0)}}/><div className="tabs">{[['all','全部'],['success','成功'],['failed','未通关']].map(([k,v])=><button className={filter===k?'selected':''} key={k} onClick={()=>{setFilter(k);setOffset(0)}}>{v}</button>)}</div></div><div className="episode-scroll">{episodes.length?episodes.map(e=><button className={'episode-item '+(chosen.includes(e.id)?'selected':'')} key={e.id} onClick={()=>toggle(e.id)}><div className="episode-check">{chosen.includes(e.id)&&<Check size={12}/>}</div><div><strong>Episode #{e.id}</strong><span>{num(e.steps)} steps · Worker {e.worker_id}</span></div><div className="episode-result"><b>{num(e.max_x)} px</b><span className={e.success?'lime':''}>{e.success?'FLAG GET':e.death?'死亡':e.reason==='interrupted'?'中途保存':'截断'}</span></div></button>):<Empty small title="尚无回合记录" detail="回合结束后自动保存完整逐步轨迹。"/>}</div><div className="pagination"><button disabled={!offset} className="icon-button" onClick={()=>setOffset(Math.max(0,offset-50))}>←</button><span>{offset+1}–{Math.min(offset+50,total)} / {total}</span><button disabled={offset+50>=total} className="icon-button" onClick={()=>setOffset(offset+50)}>→</button></div></div>
    <div className="trajectory-charts"><div className="panel"><PanelHead title="世界坐标运动轨迹" subtitle="X 从关卡左侧起算 · Y 从底部向上 · 红色 × 表示死亡终点" right={<span className="tiny-tag">GAME WORLD / px</span>}/>{traces.length?<Chart height={350} option={{tooltip:{trigger:'item'},legend:{textStyle:{color:'#94a2b6'},top:5},grid:{left:48,right:30,top:50,bottom:36},xAxis:{...axis,type:'value',min:0,name:'X'},yAxis:{...axis,type:'value',min:0,max:280,name:'Y'},series:[...traces.map((t,i)=>({name:`#${t.episode.id}`,type:'line',showSymbol:false,lineStyle:{width:2,color:colors[i]},itemStyle:{color:colors[i]},data:t.points.map(p=>[p.x_pos,p.y_pos]),markPoint:{symbol:t.episode.death?'pin':'circle',symbolSize:22,label:{formatter:t.episode.death?'×':t.episode.success?'✓':'·',fontSize:15,color:'#10151b'},itemStyle:{color:t.episode.death?'#f17e7e':colors[i]},data:[{coord:[t.points.at(-1)?.x_pos,t.points.at(-1)?.y_pos]}]}})),{type:'scatter',symbolSize:12,itemStyle:{color:'#fff',borderColor:'#b5f375',borderWidth:3},data:current?[[current.x_pos,current.y_pos]]:[]}]}}/>:<Empty title="选择一个回合，开始分析" detail="点击左侧回合。多选可对比成功与失败路径。"/>}</div>
    <div className="panel timeline"><PanelHead title="逐步轨迹回看" subtitle="此处回放坐标；原始视频请在模型演示页单独录制" right={<span className="mono">{current?`${step+1} / ${traces[0].points.length}`:'—'}</span>}/><input type="range" aria-label="回放时间轴" min={0} max={Math.max(0,(traces[0]?.points.length??1)-1)} value={step} onChange={e=>setStep(Number(e.target.value))} disabled={!traces.length}/><div className="timeline-values"><span>世界 X <b>{num(current?.x_pos)}</b></span><span>世界 Y <b>{num(current?.y_pos)}</b></span><span>Action <b>{num(current?.action)}</b></span><span>Reward <b>{num(current?.reward,2)}</b></span></div></div>
    <div className="panel"><PanelHead title="动作与逐步奖励" subtitle={traces.length?`Episode #${traces[0].episode.id} · 完整原始采样`:'选择回合后显示'}/>{traces.length?<Chart option={{tooltip:{trigger:'axis'},legend:{data:['奖励','动作'],textStyle:{color:'#8594a8'}},grid:{left:48,right:40,top:32,bottom:35},xAxis:{...axis,type:'value',name:'step'},yAxis:[{...axis,type:'value',scale:true},{...axis,type:'value',min:0,max:6}],series:[{name:'奖励',type:'line',showSymbol:false,lineStyle:{color:'#b5f375',width:1.5},data:traces[0].points.map(p=>[p.step,p.reward])},{name:'动作',type:'line',showSymbol:false,step:'end',yAxisIndex:1,lineStyle:{color:'#6bbdff',width:1,opacity:.5},data:traces[0].points.map(p=>[p.step,p.action])}]}}/>:<Empty small title="等待选择回合"/>}</div></div></div>;
}

function Demo({checkpoints,evaluations,act,pending,selected,setSelected}:{selected:string;setSelected:(id:string)=>void;checkpoints:Checkpoint[];evaluations:Job[];act:(fn:()=>Promise<any>)=>void;pending:boolean}) {
  const [world,setWorld]=useState(1),[stage,setStage]=useState(1),[device,setDevice]=useState('auto');
  const [cp,setCp]=useState('pretrained'),[episodes,setEpisodes]=useState(1),[deterministic,setDeterministic]=useState(false),[seed,setSeed]=useState(42),[maxSteps,setMaxSteps]=useState(4000),[realtime,setRealtime]=useState(true),[record,setRecord]=useState(true);
  const job=evaluations.find(e=>e.id===selected)??evaluations[0];
  const {frame,online,displayFps}=useFrame(job?.id,'evaluations');
  const state=job?.state??{};
  return <><div className="demo-layout"><div><GameView job={job} frame={frame} online={online} displayFps={displayFps} evaluation/>{state.video_available&&['completed','stopped'].includes(job?.status??'')&&<div className="panel video-panel"><PanelHead title="已录制评估视频" subtitle="真实 NES 帧 · H.264 / MP4" right={<a className="text-button" download href={`/api/evaluations/${job!.id}/video`}><ArrowDownToLine size={14}/>下载 MP4</a>}/><video controls src={`/api/evaluations/${job!.id}/video`} preload="metadata"/></div>}</div><div className="panel demo-config"><PanelHead title="模型评估配置" subtitle="独立进程，不增加训练计数"/><form onSubmit={e=>{e.preventDefault();act(async()=>{const created=await api<Job>('/evaluations',{checkpoint_id:cp,episodes,deterministic,seed,max_steps:maxSteps,realtime,record_video:record,device,world,stage});setSelected(created.id)})}}><label>策略 checkpoint<select value={cp} onChange={e=>{setCp(e.target.value);const c=checkpoints.find(c=>c.id===e.target.value);setWorld(c?.world??1);setStage(c?.stage??1)}}>{checkpoints.map(c=><option value={c.id} key={c.id}>{c.best?'★ ':''}{c.name} · {(c.algorithm??'ppo').toUpperCase()} · {c.world??1}-{c.stage??1}</option>)}</select></label><StageSelect world={world} stage={stage} onChange={(w,s)=>{setWorld(w);setStage(s)}}/><label>评估设备<select value={device} onChange={e=>setDevice(e.target.value)}><option value="auto">自动检测</option><option value="cpu">CPU</option><option value="cuda">CUDA（不可用时回退 CPU）</option></select></label><p className="field-note">可跨关卡测试现有模型；原关卡通关不代表能通过新关卡。</p><div className="form-grid"><label>评估回合<input type="number" min={1} max={20} value={episodes} onChange={e=>setEpisodes(Number(e.target.value))}/></label><label>随机种子<input type="number" min={0} value={seed} onChange={e=>setSeed(Number(e.target.value))}/></label></div><label>单局环境步数上限<input type="number" min={32} max={20000} value={maxSteps} onChange={e=>setMaxSteps(Number(e.target.value))}/></label><label>策略采样<select value={String(deterministic)} onChange={e=>setDeterministic(e.target.value==='true')}><option value="true">确定性策略（argmax）</option><option value="false">随机策略（按分布采样）</option></select></label><label className="checkbox-label"><input type="checkbox" checked={realtime} onChange={e=>setRealtime(e.target.checked)}/>实时演示（15 环境步 / 秒）</label><label className="checkbox-label"><input type="checkbox" checked={record} onChange={e=>setRecord(e.target.checked)}/>同时录制 MP4 视频</label><div className="profile-note"><Flag size={18}/><span>通关由模拟器 flag_get 判定。不同种子与策略模式可能得到不同结果。</span></div><button type="submit" className="button primary full" disabled={pending||!checkpoints.length||evaluations.some(busy)}><Play size={15}/>{evaluations.some(busy)?'评估正在运行':'启动独立评估'}</button>{busy(job)&&<button type="button" className="button subtle full" disabled={pending||job?.status==='stopping'} onClick={()=>act(()=>api(`/evaluations/${job!.id}/stop`,{}))}><Square size={13}/>停止评估</button>}</form></div></div>
    {job&&<div className="eval-result"><div><span>评估状态</span><Badge status={job.status}/></div><div><span>完成回合</span><strong>{num(state.episodes)} / {job.config.episodes}</strong></div><div><span>通关次数</span><strong className="lime">{num(state.flags_cleared)}</strong></div><div><span>通关率</span><strong>{state.clear_rate!=null?num(state.clear_rate*100,1)+'%':'—'}</strong></div><div><span>最远位置</span><strong>{num(state.best_max_x??state.current?.max_x)} <small>px</small></strong></div><div><span>评估环境步</span><strong>{num(state.eval_steps)}</strong></div></div>}
    {job?.error&&<div className="error-box">{job.error}<button className="text-button" onClick={()=>act(async()=>{const log=await api(`/jobs/${job.id}/log`);throw new Error(log.text.slice(-2000))})}>查看诊断</button></div>}
    <div className="panel"><PanelHead title="评估记录" subtitle="训练期间的自动评估与手动演示分别创建独立记录"/>{evaluations.length?<div className="table-scroll"><table><thead><tr><th>评估 ID</th><th>创建时间</th><th>来源 / 算法 / 关卡</th><th>策略模式</th><th>通关 / 回合</th><th>状态</th><th>查看</th></tr></thead><tbody>{evaluations.map(e=><tr key={e.id}><td className="mono">#{e.id.slice(0,8)}</td><td>{date(e.created_at)}</td><td>{e.config.parent_run?'训练自动评估':'手动演示'} · {(e.state.algorithm??'ppo').toUpperCase()} · {e.state.world??e.config.world??1}-{e.state.stage??e.config.stage??1}</td><td>{e.config.deterministic?'确定性':'随机'}</td><td>{num(e.state.flags_cleared)} / {num(e.state.episodes)}</td><td><Badge status={e.status}/></td><td><button className="text-button" onClick={()=>setSelected(e.id)}>查看<ArrowRight size={13}/></button></td></tr>)}</tbody></table></div>:<Empty small title="还没有评估记录" detail="预训练 PPO 与本地 PPO / A2C 权重均可独立评估。"/>}</div></>;
}

createRoot(document.getElementById('root')!).render(<App/>);
