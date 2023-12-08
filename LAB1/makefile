# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= sobel_orig sobel1_loop_interchange sobel1_test sobel2_unroll sobel3_inline sobel4_unroll2 sobel4_unroll4 sobel4_unroll8 sobel4_unroll16 sobel5_fusion sobel5_unroll2-fusion sobel5_unroll2-fusion-inter sobel5_unroll4-fusion sobel5_unroll4-fusion-inter sobel5_unroll8-fusion sobel5_unroll8-fusion-inter sobel5_unroll16-fusion sobel6_linvcode sobel7_subexpr2 sobel8_strength_reduction sobel9 sobel10_compiler_help 

#This is the compiler to use
CC = icc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -Wall -fast

CFFLAGS = -Wall -O0

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm

# make all will create all executables
all: $(EXECUTABLES)
	
# This is the rule to create any executable from the corresponding .c 
# file with the same name.
%: %.c
	$(CC) $(CFLAGS) $< -o $@_fast $(LDFLAGS) -qopt-report-file:test/report/report_$@
	$(CC) $(CFFLAGS) $< -o $@ $(LDFLAGS) 
# make clean will remove all executables, jpg files and the 
# output of previous executions.

clean:
	rm -f $(EXECUTABLES) manipulator *.jpg output_sobel.grey *_fast
	rm -rf test

# make image will create the output_sobel.jpg from the output_sobel.grey. 
# Remember to change this rule if you change the name of the output file.
image: output_sobel.grey
	convert -depth 8 -size 4096x4096 GRAY:output_sobel.grey output_sobel.jpg 

init:
	mkdir test
	mkdir test/report
	chmod 777 script.sh
	gcc -Wall -g manipulator.c -o manipulator -lm
	
run: 
	./script.sh sobel_orig
	./manipulator test/time_sobel_orig
	./script.sh sobel_orig_fast
	./script.sh sobel1_loop_interchange
	./manipulator test/time_sobel1_loop_interchange
	./script.sh sobel1_loop_interchange_fast
	./manipulator test/time_sobel1_loop_interchange_fast
	./manipulator test/time_sobel_orig_fast
	./script.sh sobel1_test
	./manipulator test/time_sobel1_test
	./script.sh sobel1_test_fast
	./manipulator test/time_sobel1_test_fast
	./script.sh sobel2_unroll
	./manipulator test/time_sobel2_unroll
	./script.sh sobel2_unroll_fast
	./manipulator test/time_sobel2_unroll_fast
	./script.sh sobel3_inline
	./manipulator test/time_sobel3_inline
	./script.sh sobel3_inline_fast
	./manipulator test/time_sobel3_inline_fast
	./script.sh sobel4_unroll2
	./manipulator test/time_sobel4_unroll2
	./script.sh sobel4_unroll2_fast
	./manipulator test/time_sobel4_unroll2_fast
	./script.sh sobel4_unroll4
	./manipulator test/time_sobel4_unroll4
	./script.sh sobel4_unroll4_fast
	./manipulator test/time_sobel4_unroll4_fast
	./script.sh sobel4_unroll8
	./manipulator test/time_sobel4_unroll8
	./script.sh sobel4_unroll8_fast
	./manipulator test/time_sobel4_unroll8_fast
	./script.sh sobel4_unroll16
	./manipulator test/time_sobel4_unroll16
	./script.sh sobel4_unroll16_fast
	./manipulator test/time_sobel4_unroll16_fast
	./script.sh sobel5_fusion
	./manipulator test/time_sobel5_fusion
	./script.sh sobel5_fusion_fast
	./manipulator test/time_sobel5_fusion_fast
	./script.sh sobel5_unroll2-fusion
	./manipulator test/time_sobel5_unroll2-fusion
	./script.sh sobel5_unroll2-fusion_fast
	./manipulator test/time_sobel5_unroll2-fusion_fast
	./script.sh sobel5_unroll2-fusion-inter
	./manipulator test/time_sobel5_unroll2-fusion-inter
	./script.sh sobel5_unroll2-fusion-inter_fast
	./manipulator test/time_sobel5_unroll2-fusion-inter_fast
	./script.sh sobel5_unroll4-fusion
	./manipulator test/time_sobel5_unroll4-fusion
	./script.sh sobel5_unroll4-fusion_fast
	./manipulator test/time_sobel5_unroll4-fusion_fast
	./script.sh sobel5_unroll4-fusion-inter
	./manipulator test/time_sobel5_unroll4-fusion-inter
	./script.sh sobel5_unroll4-fusion-inter_fast
	./manipulator test/time_sobel5_unroll4-fusion-inter_fast
	./script.sh sobel5_unroll8-fusion
	./manipulator test/time_sobel5_unroll8-fusion
	./script.sh sobel5_unroll8-fusion_fast
	./manipulator test/time_sobel5_unroll8-fusion_fast
	./script.sh sobel5_unroll16-fusion
	./manipulator test/time_sobel5_unroll16-fusion
	./script.sh sobel5_unroll16-fusion_fast
	./manipulator test/time_sobel5_unroll16-fusion_fast
	./script.sh sobel6_linvcode
	./manipulator test/time_sobel6_linvcode
	./script.sh sobel6_linvcode_fast
	./manipulator test/time_sobel6_linvcode_fast
	./script.sh sobel7_subexpr2
	./manipulator test/time_sobel7_subexpr2
	./script.sh sobel7_subexpr2_fast
	./manipulator test/time_sobel7_subexpr2_fast
	./script.sh sobel8_strength_reduction
	./manipulator test/time_sobel8_strength_reduction
	./script.sh sobel8_strength_reduction_fast
	./manipulator test/time_sobel8_strength_reduction_fast
	./script.sh sobel9
	./manipulator test/time_sobel9
	./script.sh sobel9_fast
	./manipulator test/time_sobel9_fast
	./script.sh sobel10_compiler_help
	./manipulator test/time_sobel10_compiler_help
	./script.sh sobel10_compiler_help_fast
	./manipulator test/time_sobel10_compiler_help_fast

