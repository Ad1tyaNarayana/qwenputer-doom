import {loadFont as loadAnton} from '@remotion/google-fonts/Anton';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const display = loadAnton('normal', {weights: ['400'], subsets: ['latin']}).fontFamily;
export const mono = loadMono('normal', {weights: ['400', '700'], subsets: ['latin']}).fontFamily;

export const C = {
	bg: '#07090b',
	panel: '#0c1116',
	red: '#d3261c',
	redDeep: '#7a0f0a',
	amber: '#ffb52e',
	green: '#9ff2c4',
	text: '#f2eee8',
	dim: '#8a929a',
	line: '#1e2830',
};

export const FPS = 30;

// Scene boundaries in frames.
export const S = {
	hook: 0,
	qwenputer: 60,
	pipeline: 150,
	footage: 255,
	frame: 555,
	end: 640,
	total: 760,
};
