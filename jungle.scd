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
SynthDef(\kick, {
	var env = Env.perc();
	Out.ar(0, SinOsc.ar(440) * )
}).add;

x = Synth(\kick);
)