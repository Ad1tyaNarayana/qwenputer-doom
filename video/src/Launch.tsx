import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile} from 'remotion';
import {CRT} from './fx';
import {EndCard, FinalFrame, Footage, Hook, Pipeline, Qwenputer} from './scenes';
import {C, S} from './theme';

const scenes = [
	{from: S.hook, to: S.qwenputer, C: Hook},
	{from: S.qwenputer, to: S.pipeline, C: Qwenputer},
	{from: S.pipeline, to: S.footage, C: Pipeline},
	{from: S.footage, to: S.frame, C: Footage},
	{from: S.frame, to: S.end, C: FinalFrame},
	{from: S.end, to: S.total, C: EndCard},
];

export const Launch: React.FC = () => (
	<AbsoluteFill style={{background: C.bg}}>
		{scenes.map(({from, to, C: Scene}) => (
			<Sequence key={from} from={from} durationInFrames={to - from}>
				<Scene len={to - from} />
			</Sequence>
		))}
		<CRT />
		<Audio src={staticFile('soundtrack.wav')} />
	</AbsoluteFill>
);
