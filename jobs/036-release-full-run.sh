# release 034 (it started earlier and waits on the cancelled 033 preview) once the 4k preview is done
. jobs/lib.inc
wait_for 035-lightseed-v5-preview4k
touch $W/done/033-lightseed-v5-preview.ok
echo released
