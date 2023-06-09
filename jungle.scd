// increasing server memory is recommended for better performance
// there are often ~3500 ugens at once
o = Server.local.options;
o.memSize = 2.pow(20);
s.reboot;


s.prepareForRecord;

s.record;

s.stopRecording;

(

// synthdef based on isolated amen ride
SynthDef(\ride,{|freq_ratio=1, attackTime=0.1, releaseTime=0.5, amp=1.0, pan=0.0|
var amps = [-32.949, -35.036, -34.114, -31.87, -33.709, -36.305, -36.143, -35.805, -30.095, -34.574, -32.113, -34.575].dbamp;
var harmonics = [2444.02, 3127.7, 3197.68, 3332.26, 3348.41, 4721.16, 4748.07, 5167.97, 5587.87, 5604.02, 6093.9, 6761.43];
var partial_env_amps = [0.359, 0.412, 0.479, 0.539, 0.598, 0.653, 0.703, 0.756, 0.808, 0.853, 0.887, 0.931, 0.948, 0.97, 0.982, 0.989, 0.99, 0.986, 0.986, 0.991, 0.996, 1.0, 0.999, 0.992, 0.972, 0.959, 0.941, 0.922, 0.892, 0.848, 0.796, 0.75, 0.7, 0.635, 0.584, 0.547, 0.519, 0.493, 0.48, 0.477, 0.473, 0.473, 0.481, 0.494, 0.505, 0.519, 0.53, 0.536, 0.544, 0.546, 0.548, 0.545, 0.548, 0.543, 0.542, 0.539, 0.536, 0.525, 0.516, 0.51, 0.491, 0.468, 0.446, 0.424, 0.398, 0.368, 0.336, 0.302, 0.27, 0.239, 0.204];
	// create the overall envelope specified by the user to create the "shape" of the note
	var overall_env = Env.perc(attackTime, releaseTime).kr();
	// slightly modified envelope for noise
	var noise_env = Env.perc(attackTime/1.5, releaseTime/1.5).kr();
	var duration = attackTime + releaseTime;
	// this envelope gets rid of any popping from freeing a synth before its amplitudes have gone to 0
	var no_pop_env = EnvGen.kr(Env.new(levels: [0, 1, 0], times: [duration*2], curve: 'step'), doneAction: 2);
	var num_partials = harmonics.size();
	var partial_env = EnvGen.kr(Env(partial_env_amps, times: duration/partial_env_amps.size));
	// create an array of num_partials SinOscs, with the i-th index corresponding
	// to the (i+1)-th partial
	var ride = Array.fill(num_partials, {|i|
		var f = harmonics[i];
		SinOsc.ar(
			freq: f*freq_ratio, mul: amps[i] );
	});

	// improve the percussiveness of the sound by adding some HP-filtered noise
	var noise = WhiteNoise.ar();
	noise = HPF.ar(noise, 4000)/10*noise_env;
	ride = Mix.ar(ride)*partial_env + noise;  // combine the array of partials into one audio stream
	ride = ride * amp * overall_env * no_pop_env;
	ride = ride/4;
	Out.ar(0, // output to the first bus
		Pan2.ar( // pan and wrap in overall envelope
			ride, pos:pan // pan according to user-supplied argument
		)
	)
}).add();

//Synth(\ride, [\freq_ratio, 1, \attackTime, 0.01, \releaseTime, 1]);

// synthdef based on isolated  amen kick/bass drum
SynthDef(\kick, {|freq_ratio=1, attackTime=0.001, releaseTime=0.6, amp=1.0, pan=0.0, noise_amp=1, out=0|
var amps = [-21.136, -29.981, -15.26, -23.604, -24.741, -20.489, -31.786, -31.053, -32.427, -34.03, -33.197].dbamp;
var harmonics = [69.98, 91.52, 118.43, 156.12, 177.65, 193.8, 226.1, 242.25, 263.78, 306.85, 382.21];
var partial_env_amps = [0.582, 0.69, 0.79, 0.876, 0.941, 0.986, 1.0, 0.986, 0.946, 0.884, 0.806, 0.717, 0.623, 0.53, 0.43, 0.352, 0.285, 0.197, 0.164, 0.118, 0.102, 0.09, 0.077, 0.066, 0.059, 0.056, 0.054, 0.047, 0.045, 0.043, 0.04, 0.036, 0.033, 0.029, 0.026, 0.022, 0.018, 0.015, 0.012];
	// create the overall envelope specified by the ukser to create the "shape" of the note
	var overall_env = Env.perc(attackTime, releaseTime).kr();
	// slightly modified envelope for noise
	var noise_env = Env.perc(attackTime/1.5, releaseTime/1.5).kr();
	var duration = attackTime + releaseTime;
	// this envelope gets rid of any popping from freeing a synth before its amplitudes have gone to 0
	var no_pop_env = EnvGen.kr(Env.new(levels: [0, 1, 0], times: [duration*2], curve: 'step'), doneAction: 2);
	var num_partials = harmonics.size();
	var partial_env = EnvGen.kr(Env(partial_env_amps, times: duration/partial_env_amps.size));
	// create an array of num_partials SinOscs, with the i-th index corresponding
	// to the (i+1)-th partial

	var kick = Array.fill(num_partials, {|i|
		var f = harmonics[i];
		// create new envelope based on sampled information the array of amplitudes for a given partial are uniformly spaced,
		// so we can use a scalar value for the time to create an envelope that lasts a total of duration seconds.
		SinOsc.ar(
			freq: f*freq_ratio, mul: amps[i]);
	});

	// improve the percussiveness of the sound by adding some LP-filtered noise
	var noise = BrownNoise.ar();
	noise = LPF.ar(noise, 250)*6*noise_amp;
	kick = Mix.ar(kick)*partial_env;
	kick = kick*10+noise*noise_env*noise_env;
	kick = kick * amp * overall_env * no_pop_env;
	kick = kick/15;

	Out.ar(out, // output to the first bus
		Pan2.ar( // pan and wrap in overall envelope
			kick, pos: pan // pan according to user-supplied argument
		)
	)
}).add();

// Synth(\kick);

// synthdef based on amen kick/bass drum
SynthDef(\snare, {|freq_ratio=1, attackTime=0.01, releaseTime=0.5, amp=1.0, pan=0.0|
var amps = [-22.04, -29.097, -19.754, -26.962, -16.032, -19.528, -26.167, -28.874, -28.062, -28.321, -29.021, -28.224, -28.716, -27.695, -28.625].dbamp;
var harmonics = [64.6, 236.86, 269.17, 333.76, 592.16, 947.46, 1894.92, 1970.29, 2002.59, 3154.61, 3736.01, 4629.64, 4920.34, 5124.9, 5211.04];
var partial_env_amps = [0.767, 0.857, 0.931, 0.979, 1.0, 0.995, 0.966, 0.917, 0.849, 0.772, 0.689, 0.605, 0.513, 0.44, 0.349, 0.269, 0.229, 0.194, 0.161, 0.124, 0.103, 0.086, 0.07, 0.054, 0.041];
	// create the overall envelope specified by the user to create the "shape" of the note
	var overall_env = Env.perc(attackTime, releaseTime).kr();
	var noise_env = Env.perc(attackTime, releaseTime).kr();
	var duration = attackTime + releaseTime;
	var no_pop_env = EnvGen.kr(Env.new(levels: [0, 1, 0], times: [duration*2], curve: 'step'), doneAction: 2);
	var num_partials = harmonics.size();
	var partial_env = EnvGen.kr(Env(partial_env_amps, times: duration/partial_env_amps.size));
	var snare = Array.fill(num_partials, {|i|
		var f = harmonics[i];
		SinOsc.ar(
			freq: f*freq_ratio, mul: amps[i]);
	});

	// add noise to imitate initial hit
	var noise = WhiteNoise.ar();
	noise = HPF.ar(noise, 500);
	snare = Mix.ar(snare)*partial_env;
	snare = snare*4*noise_env+noise*noise_env;
	snare = snare * amp * overall_env * no_pop_env;  // combine the array of partials into one audio stream
	snare = snare/7;
	//snare = LPF.ar(snare, 6000, 1);
	Out.ar(0, // output to the first bus
		Pan2.ar( // pan and wrap in overall envelope
			snare*amp, pos:pan // pan according to user-supplied argument
		)
	)
}).add();

//Synth(\snare);

SynthDef(\crash, {|freq_ratio=1, attackTime=0.1, releaseTime=1, amp=1.0, pan=0.0|
var amps = [-36.226, -35.915, -35.481, -35.133, -30.474, -36.17, -33.159, -35.091, -34.793, -31.982, -35.927, -32.5, -35.773, -35.727, -35.645, -35.511, -35.191, -34.066, -34.606, -34.348, -35.374, -32.744].dbamp;
var harmonics = [2923.13, 3030.8, 3170.76, 3192.3, 3348.41, 3386.1, 3396.86, 3628.34, 3757.54, 3843.68, 3886.74, 3919.04, 4085.92, 4312.02, 4495.06, 4721.16, 4742.69, 4839.59, 5237.95, 5394.07, 5835.5, 5851.65];
var partial_env_amps = [0.102, 0.113, 0.126, 0.139, 0.152, 0.168, 0.194, 0.219, 0.257, 0.316, 0.366, 0.419, 0.471, 0.538, 0.602, 0.66, 0.725, 0.77, 0.815, 0.846, 0.87, 0.909, 0.936, 0.951, 0.96, 0.971, 0.993, 1.0, 0.997, 0.993, 0.975, 0.948, 0.921, 0.884, 0.832, 0.77, 0.718, 0.66, 0.601, 0.558, 0.511, 0.475, 0.449, 0.408, 0.395, 0.391, 0.385, 0.378, 0.359, 0.351, 0.339, 0.324, 0.305, 0.278, 0.261, 0.241, 0.22, 0.194, 0.164, 0.143];
	// create the overall envelope specified by the user to create the "shape" of the note
	var overall_env = Env.perc(attackTime, releaseTime).kr();
	var noise_env = Env.perc(attackTime, releaseTime*5).kr();
	var duration = attackTime + releaseTime;
	var no_pop_env = EnvGen.kr(Env.new(levels: [0, 1, 0], times: [duration*2], curve: 'step'), doneAction: 2);

	var num_partials = harmonics.size();
	var partial_env = EnvGen.kr(Env(partial_env_amps, times: duration/partial_env_amps.size));
	// create an array of num_partials SinOscs, with the i-th index corresponding
	// to the (i+1)-th partial
	var crash = Array.fill(num_partials, {|i|
		var f = harmonics[i];
		SinOsc.ar(
			freq: f*freq_ratio, mul: amps[i]);
	});

	// add some high pass filtered whitenoise to fill out the sound
	var noise = HPF.ar(WhiteNoise.ar(), freq: 3000);

	crash = Mix.ar(crash)*partial_env;
	crash = crash*4+(noise*noise_env)/5; // combine the array of partials into one audio stream
	crash = crash * amp * overall_env * no_pop_env;
	Out.ar(0, // output to the first bus
		Pan2.ar( // pan and wrap in overall envelope
			crash,
			pos:pan // pan according to user-supplied argument
		)
	)
}).add();

Synth(\crash);

// effect taken from stealthissound.scd
SynthDef(\choruscompresseffect, {|in = 21 out = 0 gate = 1|
	var source = In.ar(in,2);
	var chorus;
	var env = Linen.kr(gate, 0.1, 1, 0.1, 2);

	chorus= Splay.ar(Array.fill(4,{
		var maxdelaytime= rrand(0.005,0.02);

		DelayC.ar(source[0], maxdelaytime,LFNoise1.kr(Rand(0.1,0.6),0.25*maxdelaytime,0.75*maxdelaytime) )
	}));

	chorus = Compander.ar(4*(source + chorus),source,0.4,1,4);

	Out.ar(out, chorus);

}).add;

// effect taken from stealthissound.scd
SynthDef(\filtereffect, {|out =0 gate= 1|
	var source = In.ar(out,2);
	var env = Linen.kr(gate, 0.0, 1, 0.1, 2);
	var output;

	output = BLowPass.ar(BLowPass.ar(source,SinOsc.ar(0.25,0,5000,5500),0.4),(200+(5000*SinOsc.ar(4.01,0.5,0.5,0.5))),0.6);

	//output = BLowPass.ar(source,SinOsc.ar(0.275,0,5000,5500),0.1);

	XOut.ar(out, env, output * env);

}).add;
)

(
var kick_pitch, drone1, drone2, drone3, reps;

// helper function to streamline creation of Pbinds
~rhythm = {|instrument, durs, amp=#[1], freqs=1, noise_amp=1|
	Pbind(\instrument, instrument,
		\amp, Pseq(amp, inf),
		\dur, Pseq(durs, 1),
		\freq_ratio, freqs,
		\noise_amp, noise_amp
	)
};

// 136 bpm - original tempo of amen break
~unit = 138/60;
t = TempoClock(~unit);

// pitch of kick is around an A#2, which will be useful for creating melodic elements using the kick synth
~kick_pitch = 46.midicps;
// this progression is supposed to sound like Gary Numan's M.E.
drone1 = 53.midicps/~kick_pitch;
drone2 = 52.midicps/~kick_pitch;
drone3 = 48.midicps/~kick_pitch;
~drone_pitches = Pseq([Pseq([drone1], 10), Pseq([drone2], 10), Pseq([drone3], 20)], 2);

// low pitch drone
~drone_pitches2 = Prand([Pseq([0.4], 20), Pseq([0.45], 20), Pseq([0.6], 20)], 4);

// mix of higher and lower pitch drones
~drone_pitches3 = Prand([Pseq([0.5], 20), Pseq([2], 20), Pseq([3], 20)], 4);


~drone1 = {|reps=1|
	Pfx(
		Pn(Pbind(\instrument, \kick, \noise_amp, 0, \out, 21, \amp, 0.15, \freq_ratio, ~drone_pitches, \dur, 1/5), reps),
		\choruscompresseffect
	);
};

~drone2 = {|reps=1|
	Pfx(
		Pn(Pbind(\instrument, \kick, \noise_amp, 0, \out, 21, \amp, 0.1, \freq_ratio, ~drone_pitches2, \dur, 1/5), reps),
		\choruscompresseffect
	);
};

~drone3 = {|reps=1|
	Pfx(
		Pn(Pbind(\instrument, \kick, \noise_amp, 0, \out, 21, \amp, 0.07, \freq_ratio, ~drone_pitches3, \dur, 1/5), reps),
		\choruscompresseffect
	);
};

// main amen break with a few different ride settings, allows for different tempos via rel_dur
// random=true enables some alternate variations
// final_snare makes it so the song can end without a lingering snare hit
~main_break = {|amp=1, ride_pat=1, rel_dur=1, final_snare=1, random=true|
	var m1, m3, m4;
	var ghost_amp = 0.5;
	// enable some snare variation
	var main_snare = if (random, {  Prand([3/4, Pseq([1/4, 1/4, 1/4])]) },{ 3/4 });
	// enable some bass drum variation
	var main_bass = if (random, {  Prand([Pseq([1/2,1/2]), Pseq([1/4, 1/4, 1/4, 1/4])]) },{ Pseq([1/2,1/2]) });

	// select between two ride patterns
	var ride = switch (ride_pat)
    {1} { ~rhythm.(\ride, rel_dur*[1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp])}
	{2} { ~rhythm.(\ride, rel_dur*[Rest(1/2), 1, 1, 1, 1/2]) };

	// pattern for first and second measure of amen break
	m1 = Ppar([
		ride,
		~rhythm.(\snare, rel_dur*[Rest(1), main_snare, 1/4, Rest(1/4), 1/4, Rest(1/2), 3/4, 1/4], amp*[0, 1, ghost_amp, 0, ghost_amp, 0, 1, ghost_amp/2]),
		~rhythm.(\kick, [main_bass, Rest(3/2), 1/4, 1/4, Rest(1)], [amp])
	]);

	// pattern for third measure of amen break
	m3 = Ppar([
		ride,
		~rhythm.(\snare, rel_dur*[Rest(1), main_snare, 1/4, Rest(1/4), 1/4, Rest(1), 1/2], amp*[0,1,ghost_amp, 0, ghost_amp, 0, 1]),
		~rhythm.(\kick, rel_dur*[main_bass, Rest(3/2), 1/2, Rest(1)], [amp]),
	]);

	// pattern for fourth measure of amen break
	m4 = Ppar([
		ride,
		~rhythm.(\crash, rel_dur*[Rest(5/2), 3/2], [amp]),
		~rhythm.(\snare, rel_dur*[Rest(1/4), 1/4, Rest(1/2), main_snare, 1/4, Rest(1/4), 1/4, Rest(1), 1/2],
			amp*[0,ghost_amp,0,1,ghost_amp, 0, ghost_amp, 0, final_snare]),
		~rhythm.(\kick, rel_dur*[Rest(1/2), 1/4, 1/4, Rest(3/2), 1/2, Rest(1)], [amp])
	]);

	Pseq([m1,m1,m3,m4]);
};

// alternate latin-style break
~latin_break = {|amp=1|
	Pn(Ppar([
	~rhythm.(\ride, [1/2, 1/2,1/4,1/4, Rest(1/4), 1/2, 1/2, 1/4, 1/4, Rest(1/4), 1/2], [amp*1.5]),
	~rhythm.(\snare, [Rest(1/2), 1/2, Rest(1/2), 1/2, Rest(1/2), 1/2, Rest(1/2), 1/2], [amp]),
	~rhythm.(\kick, [1/2, Rest(1/2), 1/4, 1/4, Rest(1/4), 1/2, 1/4, Rest(1/4), 1/4, 1/4, Rest(3/4)], [amp])
	]), 4);
};

// groove section
~groove = {|amp=0.5|
	var kick = ~rhythm.(\kick, [1/2,1/4,1/4, 1/2,1/4,1/4, 1/2,1/4,1/4, 1/2,1/4,1/4], amp*[1,0.5,0.5]);
	var groove =
	Pn(Ppar([
			~rhythm.(\ride, [Rest(1.5), 1/4, 1/4, Rest(1/4), 1/4, 1/4, Rest(3/4)], [0.5]),
			~rhythm.(\snare, [Rest(1), 1, Rest(1/4), 1/4, Rest(1/2), 1], [0.5]),
			kick
		]), 4);
	var groove_and_bell =
	Ppar([
		groove,
		Pn(~rhythm.(\kick, [1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 3/4, 1/4, Rest(3/4), 1/4],
			amp: amp*[1, 0.5, 0.5, 1, 0.5, 0.5, 1, 1, 1],
			freqs: Pseq([2, 2, 2, 1.6, 1.6, 1.6, 1.4, 1.4, 0, 1.4]), noise_amp: 0), 4)
	]);
	Pseq([
		// start with kick and add in snare, ride
		kick,
		kick,
		Ppar([
			~rhythm.(\snare, [Rest(1), 1, Rest(1), 1], [0.5]),
			kick
		]),
		Ppar([
			~rhythm.(\snare, [Rest(1), 1, Rest(1/4), 1/4, Rest(1/2), 1], [0.5]),
			kick
		]),
		groove,
		Pn(Pfx(Pfx(groove, \choruscompresseffect), \filtereffect), 2),
		~rhythm.(\snare, [2, 2, 1, 1], [0.5]),
		~rhythm.(\kick, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5], freqs: 2, noise_amp: 0),
		~rhythm.(\kick, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5], freqs: 1, noise_amp: 0),
		Pn(groove, 2),
		Pn(groove_and_bell, 4)
	])
};

// second to last section
~breakdown = {|final_reps=4, amp=0.5|
	var final_kick = ~rhythm.(\kick, [1/4, Rest(1/2), 1/4, Rest(1), 1/4,1/4, Rest(1/4), 1/2, 1/2, 1/4], [1]);
	// final groove
	var final_section =
	Pn(
		Ppar([
			~rhythm.(\crash, [1, Rest(3)], [amp]),
			~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp]),
			~rhythm.(\snare, [Rest(1/4),1/4,1/4, Rest(1/4), 1/2, 1/2, 1/4, Rest(1.75)], amp: [amp],
				freqs: Pseq([1,1,1.1, 1, 1, 1.5, 1.6, 1], 1)),
			final_kick
		]),
		4);
	Pseq([
		// call and response section
		Ppar([
			~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp]),
			~rhythm.(\snare, [Rest(1), 1/2, 1/2, Rest(2)], [amp]),
			~rhythm.(\kick, [1/2, Rest(1/4), 1/4, Rest(2.5)], [amp]),
		]),
		Ppar([
			~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp]),
			~rhythm.(\snare, [1/2, Rest(3.5)], [amp]),
			~rhythm.(\kick, [Rest(1/4), 1/4, Rest(1/4), 1/4, 1/2, 1/2, Rest(2)], [amp]),
		]),
		Ppar([
			~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp]),
			~rhythm.(\snare, [Rest(1/4),1/4,1/4, Rest(1/4), 1/2, 1/2, Rest(2)], [amp]),
			~rhythm.(\kick, [1/4, Rest(1/2), 1/4, Rest(2.5)], [amp]),
		]),
		Ppar([
			~rhythm.(\ride, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2], [amp]),
			~rhythm.(\snare, [1/4, Rest(1/2), 1/4, Rest(2.5)], [amp]),
			~rhythm.(\kick, [Rest(1/4),1/4,1/4, Rest(1/4), 1/2, 1/2, Rest(2)], [amp]),
		]),
		Pseq([
			~rhythm.(\snare, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5]),
			~rhythm.(\kick, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5], freqs: 1, noise_amp: 0),
			~rhythm.(\snare, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5]),
			~rhythm.(\kick, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5], freqs: 1, noise_amp: 0),
			~rhythm.(\snare, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5],),
			~rhythm.(\kick, [1/6,1/6,1/6,1/6,1/6,1/6], amp*[1,0.5,0.5,0.5,0.5,0.5], freqs: 1.5, noise_amp: 0),
			~rhythm.(\kick, [1/3, 1/3], [1,0.5], [amp, amp*0.5], freqs: 2, noise_amp: 0),
			~rhythm.(\kick, [1/3, 1/3], [1,0.5], [amp, amp*0.5], freqs: 1.5, noise_amp: 0),
			~rhythm.(\kick, [1/3, 1/3], [1,0.5], [amp, amp*0.5], freqs: 1, noise_amp: 0),
		]),
		final_section,
		Pn(Ppar([final_section, ~drone2.()]), final_reps)], 1)
};

// number of times to repeat various sections
reps = 4;

// build up the full song from the elements defined above
Pseq(
	[
		Pn(~main_break.(0.5, random: false), 1),
		Pn(Ppar([~main_break.(0.5), ~drone2.()]), reps),
		Pn(Ppar([~main_break.(0.5, ride_pat: 2), ~drone1.()]), reps),
		Pn(Ppar([~latin_break.(0.5), ~drone1.()]), reps),
		Pn(Ppar([~latin_break.(0.4), ~main_break.(0.4, ride_pat: 2), ~drone1.()]), reps),
		Ppar([Pn(Ppar([~main_break.(0.5, ride_pat: 1)]), 3), Pn(~main_break.(0.5, ride_pat: 1, rel_dur: 3), 1)]),
		Ppar([Pn(Ppar([~main_break.(0.5, ride_pat: 1), ~drone2.()]), 3), Pn(~main_break.(0.5, ride_pat: 1, rel_dur: 3), 1)]),
		~groove.(),
		~breakdown.(),
		Pn(~main_break.(0.5, final_snare: 0, random: false), 1)
	]
).play(t);
)
