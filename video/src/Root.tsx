import React from 'react';
import {Composition} from 'remotion';
import {Launch} from './Launch';
import {FPS, S} from './theme';

export const Root: React.FC = () => (
	<>
		<Composition id="LaunchVertical" component={Launch} durationInFrames={S.total} fps={FPS} width={1080} height={1920} />
		<Composition id="LaunchLandscape" component={Launch} durationInFrames={S.total} fps={FPS} width={1920} height={1080} />
	</>
);
