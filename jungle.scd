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
SynthDef(\kick,{|out= 0 freq = 440 amp = 0.1 ringTime=10.0 releaseTime=1.0 distortion = 0.1 pan=(-0.1)|

	var impulse, filter, env;

  	impulse = Impulse.ar(0);

	filter = Ringz.ar(impulse,XLine.ar(freq,60,0.1),ringTime); // Ringing filter

	env = EnvGen.ar(Env.perc(0.001,releaseTime),doneAction:2);

	filter = (1.0-distortion)*filter + (distortion*(filter.distort));
	Out.ar(out,Pan2.ar(filter*env*amp,pan));

}).add();

SynthDef(\snare, {
	var pink_noise = PinkNoise.ar();
	var env = EnvGen.ar(Env.perc(0.01, 0.3))*1;
    Out.ar(0,Pan2.ar(pink_noise*env, 0, 1));
}).add();

SynthDef(\hihat, {arg out = 0, amp = 0.5, att = 0.01, rel = 0.2, ffreq = 6000, pan = 0;
	var env, snd;
	env = Env.perc(att, rel, amp).kr(doneAction: 2);
	snd = WhiteNoise.ar;
	snd = HPF.ar(in: snd, freq: ffreq, mul: env);
	Out.ar(out, Pan2.ar(snd, pan));
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
	~rhythm.(\hihat, [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2]),
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

//ePbind(\instrument, \snare, \dur, Prand([Pseq([1/256],256), Pseq([1/128], 128)], 1)).play;
FreqScope.new;

Pseq([m1, m1, m3, m4], inf).play(t);
//Pseq([m1, m1, m3, m4], inf).play(t2);
//Pseq([m1, m1, m3, m4], inf).play(t3);
//Pseq([m1, m1, m3, m4], inf).play(t4);
)

SynthDef("snare", {
	var pink_noise = PinkNoise.ar();
	var env = EnvGen.ar(Env.perc(0.01, 0.3))*1;
    Out.ar(0,Pan2.ar(pink_noise*env, 0, 1));
}).play;
