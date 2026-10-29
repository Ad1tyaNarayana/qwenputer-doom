import React from 'react';
import {AbsoluteFill, interpolate, random, useCurrentFrame} from 'remotion';
import {C, mono} from './theme';

export const CRT: React.FC = () => {
	const frame = useCurrentFrame();
	const flicker = 0.035 + random(`f${frame}`) * 0.025;
	return (
		<AbsoluteFill style={{pointerEvents: 'none'}}>
			<AbsoluteFill
				style={{
					backgroundImage:
						'repeating-linear-gradient(0deg, rgba(0,0,0,0.28) 0px, rgba(0,0,0,0.28) 1px, transparent 2px, transparent 4px)',
					mixBlendMode: 'multiply',
				}}
			/>
			<AbsoluteFill style={{background: `rgba(255,255,255,${flicker})`, mixBlendMode: 'overlay'}} />
			<AbsoluteFill
				style={{
					background: 'radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.75) 100%)',
				}}
			/>
		</AbsoluteFill>
	);
};

// Short RGB-split jitter that settles after `duration` frames.
export const Glitch: React.FC<{children: React.ReactNode; start?: number; duration?: number}> = ({
	children,
	start = 0,
	duration = 10,
}) => {
	const frame = useCurrentFrame() - start;
	const active = frame >= 0 && frame < duration;
	const amt = active ? interpolate(frame, [0, duration], [14, 0]) : 0;
	const dx = active ? (random(`gx${frame}`) - 0.5) * amt : 0;
	const clip = active ? random(`gc${frame}`) * 100 : 0;
	return (
		<div style={{position: 'relative'}}>
			{active && (
				<>
					<div style={{position: 'absolute', inset: 0, transform: `translateX(${-amt}px)`, color: C.red, opacity: 0.8, clipPath: `inset(${clip}% 0 ${Math.max(0, 70 - clip)}% 0)`}}>
						{children}
					</div>
					<div style={{position: 'absolute', inset: 0, transform: `translateX(${amt}px)`, color: '#2ee6ff', opacity: 0.6}}>
						{children}
					</div>
				</>
			)}
			<div style={{position: 'relative', transform: `translateX(${dx}px)`}}>{children}</div>
		</div>
	);
};

export const Typed: React.FC<{text: string; start: number; cps?: number; cursor?: boolean; style?: React.CSSProperties}> = ({
	text,
	start,
	cps = 22,
	cursor = true,
	style,
}) => {
	const frame = useCurrentFrame();
	const n = Math.max(0, Math.floor(((frame - start) / 30) * cps));
	const shown = text.slice(0, n);
	const blink = Math.floor(frame / 8) % 2 === 0;
	return (
		<span style={{fontFamily: mono, ...style}}>
			{shown}
			{cursor && <span style={{opacity: blink ? 1 : 0, color: C.green}}>█</span>}
		</span>
	);
};

export const fadeInOut = (frame: number, len: number, inF = 8, outF = 8) =>
	interpolate(frame, [0, inF, len - outF, len], [0, 1, 1, 0], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});
