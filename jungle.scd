/*
TODO:
1. create kick, snare, ride, crash synthdefs (easy)
2. sequence into original drum beat (Pbind)
3. go crazy with effects, sequencing, layering, tempo (Pbind)
   - cool beat, with some randomness + random
   - overlaid slow melodic drone (snare)
4. Make it sound AWESOME!!!!

https://www.youtube.com/watch?v=R0NVOcP8OGI
*/

(
SynthDef(\ride,{|freq=440, attackTime=1.0, sustainTime=1.0, releaseTime=1.5, amp=1.0, vibrato=0.5, pan=0.0|

	var harmonics = [198.88, 2343.35, 2948.65, 3026.47, 3657.71, 3839.29, 4479.18, 4652.12, 4928.82, 5119.06, 5629.24, 5707.06, 5793.53, 5897.29, 6234.53, 6269.12, 6398.82, 6424.76, 6476.65, 6493.94, 6545.82, 7436.47, 7695.88, 7903.41, 9174.53, 10030.59];

	var envs = [
		[0.19, 0.218, 0.247, 0.272, 0.293, 0.309, 0.322, 0.33, 0.343, 0.405, 0.445, 0.515, 0.571, 0.63, 0.707, 0.783, 0.861, 0.928, 0.978, 0.999, 1.0, 0.968],
		[0.375, 0.457, 0.557, 0.654, 0.744, 0.824, 0.892, 0.949, 0.984, 1.0, 0.994, 0.972, 0.93, 0.874, 0.805, 0.72, 0.639, 0.561, 0.471, 0.398, 0.335, 0.268],
		[0.724, 0.806, 0.866, 0.928, 0.978, 0.996, 1.0, 0.996, 0.989, 0.978, 0.96, 0.941, 0.912, 0.853, 0.803, 0.756, 0.707, 0.653, 0.585, 0.514, 0.445, 0.376],
		[0.612, 0.709, 0.797, 0.871, 0.934, 0.974, 1.0, 0.999, 0.98, 0.945, 0.894, 0.837, 0.765, 0.692, 0.604, 0.538, 0.478, 0.421, 0.347, 0.306, 0.256, 0.215],
		[0.824, 0.899, 0.953, 0.991, 1.0, 0.982, 0.95, 0.901, 0.845, 0.782, 0.72, 0.66, 0.604, 0.51, 0.467, 0.42, 0.384, 0.348, 0.307, 0.27, 0.228, 0.193],
		[0.775, 0.86, 0.929, 0.974, 0.999, 1.0, 0.981, 0.94, 0.88, 0.81, 0.733, 0.655, 0.578, 0.465, 0.413, 0.365, 0.314, 0.27, 0.209, 0.181, 0.155, 0.129],
		[0.609, 0.698, 0.782, 0.854, 0.917, 0.956, 0.988, 1.0, 0.99, 0.971, 0.938, 0.903, 0.858, 0.811, 0.749, 0.712, 0.655, 0.612, 0.557, 0.483, 0.42, 0.371],
		[0.463, 0.555, 0.658, 0.75, 0.834, 0.902, 0.957, 0.99, 1.0, 0.996, 0.979, 0.942, 0.888, 0.827, 0.756, 0.677, 0.6, 0.534, 0.473, 0.412, 0.355, 0.3],
		[0.915, 0.969, 1.0, 0.998, 0.97, 0.914, 0.841, 0.757, 0.671, 0.589, 0.512, 0.447, 0.387, 0.279, 0.241, 0.226, 0.207, 0.19, 0.173, 0.153, 0.134, 0.114],
		[0.882, 0.95, 0.991, 1.0, 0.983, 0.934, 0.868, 0.784, 0.694, 0.602, 0.518, 0.444, 0.38, 0.275, 0.251, 0.221, 0.187, 0.179, 0.17, 0.156, 0.141, 0.125],
		[0.65, 0.738, 0.815, 0.878, 0.93, 0.976, 0.995, 1.0, 0.99, 0.978, 0.956, 0.929, 0.895, 0.833, 0.792, 0.738, 0.672, 0.626, 0.569, 0.507, 0.439, 0.365],
		[0.65, 0.738, 0.815, 0.878, 0.93, 0.976, 0.995, 1.0, 0.99, 0.978, 0.956, 0.929, 0.895, 0.833, 0.792, 0.738, 0.672, 0.626, 0.569, 0.507, 0.439, 0.365],
		[0.764, 0.851, 0.923, 0.968, 0.998, 1.0, 0.985, 0.955, 0.907, 0.854, 0.796, 0.745, 0.691, 0.62, 0.586, 0.532, 0.495, 0.469, 0.427, 0.392, 0.348, 0.306],
		[0.764, 0.851, 0.923, 0.968, 0.998, 1.0, 0.985, 0.955, 0.907, 0.854, 0.796, 0.745, 0.691, 0.62, 0.586, 0.532, 0.495, 0.469, 0.427, 0.392, 0.348, 0.306],
		[0.802, 0.886, 0.95, 0.991, 1.0, 0.983, 0.942, 0.883, 0.811, 0.733, 0.653, 0.576, 0.504, 0.427, 0.336, 0.285, 0.243, 0.214, 0.188, 0.161, 0.131, 0.106],
		[0.802, 0.886, 0.95, 0.991, 1.0, 0.983, 0.942, 0.883, 0.811, 0.733, 0.653, 0.576, 0.504, 0.427, 0.336, 0.285, 0.243, 0.214, 0.188, 0.161, 0.131, 0.106],
		[0.924, 0.977, 1.0, 0.991, 0.954, 0.894, 0.812, 0.719, 0.629, 0.54, 0.462, 0.398, 0.345, 0.23, 0.226, 0.218, 0.209, 0.207, 0.198, 0.185, 0.174, 0.161],
		[0.924, 0.977, 1.0, 0.991, 0.954, 0.894, 0.812, 0.719, 0.629, 0.54, 0.462, 0.398, 0.345, 0.23, 0.226, 0.218, 0.209, 0.207, 0.198, 0.185, 0.174, 0.161],
		[0.911, 0.971, 0.999, 1.0, 0.97, 0.915, 0.842, 0.756, 0.666, 0.577, 0.506, 0.437, 0.377, 0.266, 0.239, 0.228, 0.214, 0.204, 0.194, 0.181, 0.165, 0.152],
		[0.911, 0.971, 0.999, 1.0, 0.97, 0.915, 0.842, 0.756, 0.666, 0.577, 0.506, 0.437, 0.377, 0.266, 0.239, 0.228, 0.214, 0.204, 0.194, 0.181, 0.165, 0.152],
		[0.911, 0.971, 0.999, 1.0, 0.97, 0.915, 0.842, 0.756, 0.666, 0.577, 0.506, 0.437, 0.377, 0.266, 0.239, 0.228, 0.214, 0.204, 0.194, 0.181, 0.165, 0.152],
		[0.552, 0.645, 0.747, 0.829, 0.901, 0.953, 0.988, 1.0, 0.997, 0.97, 0.929, 0.871, 0.804, 0.725, 0.635, 0.557, 0.487, 0.422, 0.353, 0.297, 0.237, 0.196],
		[0.789, 0.885, 0.951, 0.99, 1.0, 0.987, 0.948, 0.887, 0.814, 0.733, 0.651, 0.572, 0.496, 0.431, 0.333, 0.261, 0.237, 0.21, 0.18, 0.154, 0.128, 0.108],
		[0.735, 0.829, 0.908, 0.962, 0.993, 1.0, 0.984, 0.95, 0.897, 0.832, 0.761, 0.686, 0.609, 0.521, 0.423, 0.373, 0.326, 0.276, 0.229, 0.18, 0.156, 0.132],
		[0.451, 0.544, 0.636, 0.724, 0.812, 0.887, 0.943, 0.979, 1.0, 0.995, 0.976, 0.94, 0.886, 0.819, 0.738, 0.668, 0.594, 0.526, 0.448, 0.39, 0.342, 0.277],
		[0.865, 0.936, 0.982, 1.0, 0.998, 0.967, 0.919, 0.854, 0.78, 0.703, 0.625, 0.555, 0.489, 0.37, 0.34, 0.308, 0.268, 0.244, 0.209, 0.181, 0.162, 0.134],
	];
	// create the overall envelope specified by the user to create the "shape" of the note
	//var overall_env = Env.linen(attackTime, sustainTime, releaseTime, amp, \welch).kr();
	var duration = attackTime + sustainTime + releaseTime;
	var num_partials = envs.size();
	// create an array of num_partials SinOscs, with the i-th index corresponding
	// to the (i+1)-th partial
	var n_samples = envs[0].size();
	var ride = Array.fill(num_partials, {|i|
		// interpolate between the g3 and b3 partial amplitudes
		// using a similar method as mapValues on hw2 clipping interpolated
		// values for frequencies not between g3 and b3
		var samples = envs[i];
		var f = harmonics[i];
		// create new envelope based on sampled information, kill the envelope when done
		// the array of amplitudes for a given partial are uniformly spaced, so we can use
		// a scalar value for the time to create an envelope that lasts a total of duration seconds.
		var env = EnvGen.kr(Env(samples, times: duration/n_samples));
		p = i+1;
		samples.postln;
		SinOsc.ar(
			freq: f,

			mul: ( env ));
	});
	// note not using Done because each partial's envelope has doneAction: 2, meaning the synth gets freed
	// when the envelope ends (this is equivalent to freeing the synth, as the synth count in the bottom
	// right goes from 1s to 0s when the play call is over)
	Out.ar(0, // output to the first bus
		Pan2.ar( // pan and wrap in overall envelope
			Env.perc().ar(doneAction: 2) * Mix.ar(ride), // combine the array of partials into one audio stream
			pos:pan // pan according to user-supplied argument
		)
	)
}).add();

Synth(\ride, [\attackTime, 0.001, \sustainTime, 0.1, \releaseTime, 0.1]);
)



(
SynthDef(\kick,{|out= 0 freq = 440 amp = 0.1 ringTime=10.0 releaseTime=1.0 distortion = 0.1 pan=(-0.1)|

	var impulse, filter, env;

  	impulse = Impulse.ar(0);

	filter = Ringz.ar(impulse,XLine.ar(freq,60,0.1),ringTime); // Ringing filter

	env = EnvGen.ar(Env.perc(0.001,releaseTime),doneAction:2);

	filter = (1.0-distortion)*filter + (distortion*(filter.distort));
	Out.ar(out,Pan2.ar(filter*env*amp,pan));

}).add();

SynthDef(\snare, {arg out = 0, amp = 0.1, sinfreq = 180, att = 0.01, rel = 0.2, ffreq = 2000, pan = 0;
	var env, snd1, snd2, sum;
	env = Env.perc(att, rel, amp).kr(doneAction: 2);
	snd1 = HPF.ar(
		in: WhiteNoise.ar,
		freq: ffreq,
		mul: env
	);
	snd2 = SinOsc.ar(freq: sinfreq, mul: env);
	sum = snd1 + snd2;
	Out.ar(out, Pan2.ar(sum, pan));
}).add();

SynthDef(\hihat, {arg out = 0, amp = 0.5, att = 0.01, rel = 0.2, ffreq = 6000, pan = 0;
	var env, snd;
	env = Env.perc(att, rel, amp).kr(doneAction: 2);
	snd = WhiteNoise.ar;
	snd = HPF.ar(in: snd, freq: ffreq, mul: env);
	Out.ar(out, Pan2.ar(snd, pan));
}).add();

SynthDef(\ride, {|center_freq = 6000, amp = 1, attackT = 0.01, releaseT = 0.1, pan = 0|
	var env = Env.perc(attackT, releaseT).kr(doneAction: 2) * amp;
	var ride = Ringz.ar(WhiteNoise.ar(), center_freq);
	Out.ar(0,
		Pan2.ar(
			ride,
			pos: pan)
	);
}).add();

SynthDef(\crash, { arg amp = 1;
  var whiteNoise = PinkNoise.ar(
    mul: Env.perc(releaseTime: 4, curve: -7).kr(doneAction: 2));
  Out.ar(0, Pan2.ar(HPF.ar(whiteNoise, freq: 7040)) * 4 * amp);
}).add;
)


(
var m1, m3, m4, t2, t3,t4;
~rhythm = {|instrument, durs|
	Pbind(\instrument, instrument,
		\dur, Pseq(durs, 1)
	)
};
// 136 bpm
t = TempoClock(136/60);
t2 = TempoClock(136/60*1/2);
t3 = TempoClock(136/60*1/8);
t4 = TempoClock(136/60*2);
m1 = Ppar([
	~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2]),
	~rhythm.(\snare, [Rest(1), 3/4, 1/4, Rest(1/4), 1/4, Rest(1/2), 3/4, 1/4]),
	~rhythm.(\kick, [1/2, 1/2, Rest(3/2), 1/4, 1/4, Rest(1)])
]);

m3 = Ppar([
	~rhythm.(\hihat, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2]),
	~rhythm.(\snare, [Rest(1), 3/4, 1/4, Rest(1/4), 1/4, Rest(1), 1/2]),
	~rhythm.(\kick, [1/2, 1/2, Rest(3/2), 1/2, Rest(1)])
]);

m4 = Ppar([
	~rhythm.(\hihat, [1/2, 1/2, 1/2, 1/2, 1/2, Rest(1/2), 1/2, 1/2]),
	~rhythm.(\crash, [Rest(5/2), 3/2]),
	~rhythm.(\snare, [Rest(1/4), 1/4, Rest(1/2), 3/4, 1/4, Rest(1/4), 1/4, Rest(1), 1/2]),
	~rhythm.(\kick, [Rest(1/2), 1/4, 1/4, Rest(3/2), 1/2, Rest(1)])
]);

//Pbind(\instrument, \snare, \dur, Prand([Pseq([1/256],256), Pseq([1/128], 128)], inf)).play;
FreqScope.new;
Pseq([m1, m1, m3, m4], inf).play(t);
//Pseq([m1, m1, m3, m4], inf).play(t2);
//Pseq([m1, m1, m3, m4], inf).play(t3);
//Pseq([m1, m1, m3, m4], inf).play(t4);
)