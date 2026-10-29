import React from 'react';
import {
	AbsoluteFill,
	Img,
	OffthreadVideo,
	interpolate,
	spring,
	staticFile,
	useCurrentFrame,
	useVideoConfig,
	Easing,
} from 'remotion';
import {C, display, mono} from './theme';
import {Glitch, Typed, fadeInOut} from './fx';

const useLayout = () => {
	const {width, height} = useVideoConfig();
	return {vertical: height > width, width, height};
};

const pop = (frame: number, fps: number, delay = 0) =>
	spring({frame: frame - delay, fps, config: {damping: 14, stiffness: 160, mass: 0.6}});

const Kicker: React.FC<{children: React.ReactNode; color?: string}> = ({children, color = C.green}) => (
	<div style={{fontFamily: mono, fontSize: 30, letterSpacing: 6, color, textTransform: 'uppercase'}}>{children}</div>
);

/* ---------- 1. Hook ---------- */
export const Hook: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {vertical} = useLayout();
	const {fps} = useVideoConfig();
	const s = pop(frame, fps, 30);
	return (
		<AbsoluteFill style={{background: C.bg, justifyContent: 'center', padding: vertical ? 80 : 160, opacity: fadeInOut(frame, len, 1, 4)}}>
			<div style={{fontSize: vertical ? 50 : 56, color: C.text}}>
				<span style={{fontFamily: mono, color: C.green}}>$ </span>
				<Typed text="can a language model run DOOM?" start={2} cps={42} cursor={frame < 30} />
			</div>
			{frame >= 30 && (
				<div style={{marginTop: 50, transform: `scale(${0.6 + 0.4 * s})`, transformOrigin: 'left center'}}>
					<Glitch start={30} duration={10}>
						<div style={{fontFamily: display, fontSize: vertical ? 150 : 170, lineHeight: 0.95, color: C.red, textTransform: 'uppercase'}}>
							Not by
							{vertical ? <br /> : ' '}
							asking it.
						</div>
					</Glitch>
				</div>
			)}
		</AbsoluteFill>
	);
};

/* ---------- 2. Qwenputer ---------- */
const CPU_LAYERS = 11;
const QWEN_LAYERS = 32;

export const Qwenputer: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const {vertical, width} = useLayout();
	const pad = vertical ? 80 : 160;
	const total = CPU_LAYERS + QWEN_LAYERS;
	const barGap = 6;
	const barW = (width - pad * 2 - barGap * (total - 1)) / total;
	const t = pop(frame, fps, 0);
	return (
		<AbsoluteFill style={{background: C.bg, padding: pad, justifyContent: 'center', opacity: fadeInOut(frame, len, 1, 6)}}>
			<Kicker>by Divyansh Agrawal</Kicker>
			<Glitch start={0} duration={10}>
				<div style={{fontFamily: display, fontSize: vertical ? 200 : 220, color: C.text, lineHeight: 1, marginTop: 10, transform: `translateY(${(1 - t) * 40}px)`}}>
					QWENPUTER
				</div>
			</Glitch>
			<div style={{fontFamily: mono, fontSize: vertical ? 42 : 44, color: C.text, marginTop: 30, lineHeight: 1.35, maxWidth: 1500, opacity: interpolate(frame, [4, 14], [0, 1], {extrapolateRight: 'clamp'})}}>
				Qwen3.5-4B with a <span style={{color: C.red}}>32-bit RISC-V CPU</span> compiled directly into its weights.
			</div>
			<div style={{display: 'flex', gap: barGap, marginTop: 70, height: vertical ? 260 : 200, alignItems: 'flex-end'}}>
				{Array.from({length: total}).map((_, i) => {
					const isCpu = i < CPU_LAYERS;
					const p = interpolate(frame, [10 + i * 0.8, 18 + i * 0.8], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
					return (
						<div
							key={i}
							style={{
								width: barW,
								height: `${p * 100}%`,
								background: isCpu ? C.red : '#2a3540',
								boxShadow: isCpu ? `0 0 18px ${C.red}88` : 'none',
							}}
						/>
					);
				})}
			</div>
			<div style={{display: 'flex', justifyContent: 'space-between', fontFamily: mono, fontSize: 26, marginTop: 18, color: C.dim, opacity: interpolate(frame, [38, 48], [0, 1], {extrapolateRight: 'clamp'})}}>
				<span style={{color: C.red}}>11 compiled CPU layers</span>
				<span>Qwen's own layers →</span>
			</div>
			<div style={{fontFamily: mono, fontSize: vertical ? 32 : 30, color: C.green, marginTop: 50, opacity: interpolate(frame, [44, 54], [0, 1], {extrapolateRight: 'clamp'})}}>
				No learned weights in the CPU. Gated DeltaNet state is the RAM.
			</div>
		</AbsoluteFill>
	);
};

/* ---------- 3. Pipeline ---------- */
const STEPS = [
	{k: 'DOOM E1M1', v: 'C source · DMC strip renderer'},
	{k: 'RV32IM', v: 'bare-metal firmware · 64 KiB RAM'},
	{k: '6,342,933', v: 'self-generated trace tokens'},
	{k: 'RTX 5070 Ti', v: 'compiled circuit on Triton'},
	{k: '320 × 240', v: 'pixels read back from circuit RAM'},
];

export const Pipeline: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const {vertical} = useLayout();
	return (
		<AbsoluteFill style={{background: C.bg, padding: vertical ? 80 : 90, justifyContent: 'center', opacity: fadeInOut(frame, len, 1, 6)}}>
			<Glitch start={0} duration={10}>
				<div style={{fontFamily: display, fontSize: vertical ? 160 : 150, color: C.text, lineHeight: 1}}>
					So I fed it <span style={{color: C.red}}>DOOM.</span>
				</div>
			</Glitch>
			<div style={{display: 'flex', flexDirection: vertical ? 'column' : 'row', gap: vertical ? 22 : 18, marginTop: vertical ? 80 : 90, alignItems: vertical ? 'stretch' : 'stretch'}}>
				{STEPS.map((st, i) => {
					const p = pop(frame, fps, 14 + i * 11);
					const last = i === STEPS.length - 1;
					return (
						<React.Fragment key={st.k}>
							<div
								style={{
									flex: 1,
									opacity: p,
									transform: vertical ? `translateX(${(1 - p) * -60}px)` : `translateY(${(1 - p) * 40}px)`,
									border: `2px solid ${last ? C.red : C.line}`,
									background: C.panel,
									padding: vertical ? '26px 34px' : '26px 24px',
								}}
							>
								<div style={{fontFamily: display, fontSize: vertical ? 70 : 56, color: last ? C.red : C.amber, lineHeight: 1, whiteSpace: 'nowrap'}}>{st.k}</div>
								<div style={{fontFamily: mono, fontSize: 28, color: C.dim, marginTop: 12, lineHeight: 1.3}}>{st.v}</div>
							</div>
							{!last && (
								<div style={{alignSelf: 'center', fontFamily: mono, fontSize: 34, color: C.green, opacity: p, lineHeight: 1}}>
									{vertical ? '↓' : '→'}
								</div>
							)}
						</React.Fragment>
					);
				})}
			</div>
		</AbsoluteFill>
	);
};

/* ---------- 4. Real footage ---------- */
// The recording is 1920x1080 at 60x. Played at 4x here, so 240x real time.
const DOOM_CROP = {x: 720, y: 200, w: 1136, h: 852};
const STATS_CROP = {x: 44, y: 236, w: 620, h: 580};

const Crop: React.FC<{crop: typeof DOOM_CROP; scale: number; trimBefore: number; style?: React.CSSProperties}> = ({crop, scale, trimBefore, style}) => (
	<div style={{width: crop.w * scale, height: crop.h * scale, overflow: 'hidden', position: 'relative', ...style}}>
		<OffthreadVideo
			src={staticFile('doom-live-timelapse-60x.mp4')}
			trimBefore={trimBefore}
			playbackRate={4}
			muted
			style={{position: 'absolute', width: 1920 * scale, height: 1080 * scale, left: -crop.x * scale, top: -crop.y * scale, maxWidth: 'none'}}
		/>
	</div>
);

const RecBadge: React.FC = () => {
	const frame = useCurrentFrame();
	const on = Math.floor(frame / 15) % 2 === 0;
	return (
		<div style={{display: 'flex', alignItems: 'center', gap: 16, fontFamily: mono, fontSize: 28, color: C.text, letterSpacing: 3}}>
			<div style={{width: 20, height: 20, borderRadius: 10, background: C.red, opacity: on ? 1 : 0.25, boxShadow: `0 0 14px ${C.red}`}} />
			REAL GPU CAPTURE · 240× TIMELAPSE
		</div>
	);
};

export const Footage: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {vertical} = useLayout();
	const trimBefore = 45; // skip most of the 2 s opening hold
	const caption = interpolate(frame, [20, 35], [0, 1], {extrapolateRight: 'clamp'});
	if (vertical) {
		const s = 1000 / DOOM_CROP.w;
		return (
			<AbsoluteFill style={{background: C.bg, alignItems: 'center', justifyContent: 'center', opacity: fadeInOut(frame, len, 1, 4)}}>
				<RecBadge />
				<div style={{fontFamily: display, fontSize: 92, color: C.text, marginTop: 30, marginBottom: 40, lineHeight: 1}}>
					Rendering, <span style={{color: C.red}}>strip by strip</span>
				</div>
				<Crop crop={DOOM_CROP} scale={s} trimBefore={trimBefore} style={{border: `3px solid ${C.line}`}} />
				<div style={{display: 'flex', width: 1000, marginTop: 40, gap: 30, alignItems: 'flex-start'}}>
					<Crop crop={STATS_CROP} scale={0.95} trimBefore={trimBefore} />
					<div style={{flex: 1, fontFamily: mono, fontSize: 28, color: C.dim, lineHeight: 1.45, opacity: caption, paddingTop: 10}}>
						Checkerboard = pixels the model hasn't produced yet.
						<br />
						<br />
						<span style={{color: C.green}}>Every pixel is read out of the circuit's own recurrent RAM.</span>
					</div>
				</div>
			</AbsoluteFill>
		);
	}
	const s = 0.84;
	return (
		<AbsoluteFill style={{background: C.bg, alignItems: 'center', justifyContent: 'center', opacity: fadeInOut(frame, len, 1, 4)}}>
			<div style={{width: 1920 * s, display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 22}}>
				<RecBadge />
				<div style={{fontFamily: mono, fontSize: 26, color: C.green, opacity: caption}}>checkerboard = not generated yet</div>
			</div>
			<Crop crop={{x: 0, y: 0, w: 1920, h: 1080}} scale={s} trimBefore={trimBefore} style={{border: `3px solid ${C.line}`}} />
		</AbsoluteFill>
	);
};

/* ---------- 5. Final frame + stats ---------- */
const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

const Stat: React.FC<{big: React.ReactNode; small: string; color?: string; delay: number; vertical: boolean}> = ({big, small, color = C.text, delay, vertical}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const p = pop(frame, fps, delay);
	return (
		<div style={{opacity: p, transform: `translateY(${(1 - p) * 30}px)`}}>
			<div style={{fontFamily: display, fontSize: vertical ? 96 : 104, color, lineHeight: 1}}>{big}</div>
			<div style={{fontFamily: mono, fontSize: vertical ? 26 : 26, color: C.dim, marginTop: 6, letterSpacing: 1}}>{small}</div>
		</div>
	);
};

export const FinalFrame: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {vertical} = useLayout();
	const zoom = interpolate(frame, [0, len], [1.0, 1.06]);
	const flash = interpolate(frame, [0, 6], [1, 0], {extrapolateRight: 'clamp'});
	const tokens = interpolate(frame, [6, 34], [0, 6342933], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.exp)});
	const imgW = vertical ? 1000 : 1080;
	const stats = (
		<div style={{display: 'grid', gridTemplateColumns: vertical ? '1fr 1fr' : '1fr', gap: vertical ? '44px 30px' : 34}}>
			<Stat vertical={vertical} delay={4} big={fmt(tokens)} small="tokens generated" color={C.amber} />
			<Stat vertical={vertical} delay={12} big="0" small="mismatches vs. reference CPU" color={C.green} />
			<Stat vertical={vertical} delay={20} big="25 / 25" small="strips read from GPU RAM" />
			<Stat vertical={vertical} delay={28} big="50.7 MIN" small="on one RTX 5070 Ti" color={C.red} />
		</div>
	);
	return (
		<AbsoluteFill style={{background: C.bg, opacity: fadeInOut(frame, len, 1, 5)}}>
			<AbsoluteFill
				style={{
					flexDirection: vertical ? 'column' : 'row',
					alignItems: 'center',
					justifyContent: 'center',
					gap: vertical ? 70 : 90,
					padding: vertical ? 40 : 80,
				}}
			>
				{vertical && <Kicker color={C.red}>The first Doom frame on Qwenputer</Kicker>}
				<div style={{position: 'relative', overflow: 'hidden', border: `3px solid ${C.red}`, boxShadow: `0 0 80px ${C.redDeep}`}}>
					<Img
						src={staticFile('doom-frame.png')}
						style={{display: 'block', width: imgW, height: (imgW * 3) / 4, imageRendering: 'pixelated', transform: `scale(${zoom})`}}
					/>
					<div style={{position: 'absolute', top: 18, left: 18, fontFamily: mono, fontSize: 22, color: C.green, background: '#000a', padding: '6px 12px', letterSpacing: 2}}>
						E1M1 · VERIFIED
					</div>
				</div>
				<div style={{width: vertical ? 1000 : 560}}>{stats}</div>
			</AbsoluteFill>
			<AbsoluteFill style={{background: C.text, opacity: flash * 0.9}} />
		</AbsoluteFill>
	);
};

/* ---------- 6. End card ---------- */
export const EndCard: React.FC<{len: number}> = ({len}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const {vertical} = useLayout();
	const p = pop(frame, fps, 0);
	const sub = interpolate(frame, [14, 26], [0, 1], {extrapolateRight: 'clamp'});
	const small = interpolate(frame, [26, 40], [0, 1], {extrapolateRight: 'clamp'});
	return (
		<AbsoluteFill style={{background: C.bg, alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: vertical ? 70 : 120, opacity: fadeInOut(frame, len, 1, 12)}}>
			<Glitch start={0} duration={14}>
				<div style={{fontFamily: display, fontSize: vertical ? 300 : 300, color: C.red, lineHeight: 0.9, transform: `scale(${0.7 + 0.3 * p})`, textShadow: `0 0 60px ${C.redDeep}`}}>DOOM</div>
			</Glitch>
			<div style={{fontFamily: display, fontSize: vertical ? 100 : 100, color: C.text, marginTop: 10, opacity: sub, lineHeight: 1}}>ON QWENPUTER</div>
			<div style={{fontFamily: mono, fontSize: vertical ? 34 : 38, color: C.green, marginTop: 50, opacity: sub}}>
				github.com/Ad1tyaNarayana/qwenputer-doom
			</div>
			<div style={{fontFamily: mono, fontSize: vertical ? 22 : 22, color: C.dim, marginTop: 60, lineHeight: 1.6, maxWidth: 1400, opacity: small}}>
				One verified frame through Qwenputer's compact Triton circuit executor — not stock HF inference, not interactive play. Yet.
				<br />
				Qwenputer by Divyansh Agrawal · DMC100-Doom by Artur Bieniek · Aspire PIXO port by atc1441 · DOOM © id Software
			</div>
		</AbsoluteFill>
	);
};
