#include "../../share/share.h"


void prepare_samples(Uchar *arSamples, int sample_count, Uint arCount[BIN256])
{
	srand(0x40);
	for(int i=0; i<sample_count; i++)
	{
		int ball = rand() % BIN256;
		arSamples[i] = ball;
		arCount[ball]++ ;
	}
}

bool verify_bin_result(const Uint std[BIN256], const Uint chk[BIN256])
{
	for(int i=0; i<BIN256; i++)
	{
		if(std[i]!=chk[i])
		{
			printf("ERROR at sample index %d, correct: %d , wrong: %d\n",
				i, std[i], chk[i]);
			return false;
		}
	}
	return true;
}

struct ThreadParam_st
{
	const Uchar *arSamples;
	int sample_count;

	Uint *arCount; // arCount[256]
};

void _cpu_count_histogram(void *param) // as CPU thread function
{
	ThreadParam_st &tp = *(ThreadParam_st*)param;

	for(int i=0; i<tp.sample_count; i++)
	{
		// Note: Since x86 & x64 "inc 1" is only one instruction(inherently atomic),
		// I just do plain ++ here.
		tp.arCount[tp.arSamples[i]]++;
	}
}

void generate_histogram_cpu(const char *title, int sample_count)
{
	int i;
	Uchar *arSamples = new Uchar[sample_count]; // cpu mem
	Uint arCount_init[BIN256] = {}; // histogram init counted, the correct answer
	Uint arCount[BIN256] = {}; 

	prepare_samples(arSamples, sample_count, arCount_init);

	printf("[%s] Counting %d samples...\n", title, sample_count);

	uint64 usec_start = ps_GetOsMicrosecs64();

	if(strcmp(title, "CPU_one_thread")==0)
	{
		for(i=0; i<sample_count; i++)
		{
			arCount[arSamples[i]]++;
		}
	}
	else if(strcmp(title, "CPU_two_threads")==0)
	{
		Uint *arCountA = arCount;
		Uint arCountB[BIN256] = {};
		int partA_samples = sample_count/2;
		
		ThreadParam_st tpA = { arSamples, partA_samples, arCountA };
		ThreadParam_st tpB = { arSamples+partA_samples, sample_count-partA_samples, arCountB };

		GGT_HSimpleThread hthread = ggt_simple_thread_create(_cpu_count_histogram, &tpA);

		_cpu_count_histogram(&tpB);

		ggt_simple_thread_waitend(hthread);

		// merge count of partA and partB
		for(i=0; i<BIN256; i++)
			arCountA[i] += arCountB[i];
	}
	else
	{
		printf("ERROR: Unknown CPU title requested: %s\n", title);
		exit(1);
	}
	
	uint64 usec_end = ps_GetOsMicrosecs64();
	uint64 usec_used = usec_end - usec_start;

	// Verify CPU-counted result.
	//
	printf("Verifying... ");
	const char *errprefix = nullptr;
	bool vsucc = verify_bin_result(arCount_init, arCount);
	if(!vsucc)
		errprefix = "Error!!!";

	if(usec_used==0)
	{
		printf("%s REAL: 0 millisec\n", 
			errprefix ? errprefix : "Success.");
	}
	else
	{
		printf("%s REAL: %.5g millisec, %.5g GB/s\n", 
			errprefix ? errprefix : "Success.",
			(double)usec_used/1000,
			(double)sample_count/1000/usec_used);
	}

	delete arSamples;
}
