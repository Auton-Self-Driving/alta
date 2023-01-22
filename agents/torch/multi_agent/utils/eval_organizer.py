import os, glob, sys, re, argparse

def display_summary(startpath, exp_names):

    pattern1 = re.compile(r"ep[0-9]*rk[0-9a-zA-Z_]*.mp4")
    pattern2 = re.compile(r"__")
    f = lambda x : x[x.find("__")+2:].split(".")[0]
    skip = False

    for root, dirs, files in os.walk(startpath):
           
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        basename = os.path.basename(root)


        if (level == 3 and len(exp_names) > 0 and basename not in exp_names) :
            skip = True
        elif level <= 3:
            skip = False

        if not skip:

            if re.match(r"ep[0-9]*rk[0-9a-zA-Z_]*", basename):
                continue

            print('{}{}'.format(indent, basename), end=" ")

            if len(files) > 0:
                # print('{}{}'.format(indent, files))
                relevant_files = list(filter(pattern1.match, files))
                relevant_stubs = [f(x) for x in relevant_files]
                
                counts = dict()
                for terms in relevant_stubs:
                    counts[terms] = counts.get(terms, 0) + 1

                print(counts,"[{}]".format(len(relevant_stubs) ))

            else:
                print()
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluations organizer")
    parser.add_argument("--exps", nargs='*', 
        default=[#["7dim_nocrach_dense_no_lane","15dim_nocrach_dense_no_lane_term_tanh_squashed",
        "14dim_nocrach_dense_no_lane_term_tanh_squashed"])#,"15dim_nocrach_dense_no_lane_term_tanh_squashed_sp_30_wp_10"])

    args = parser.parse_args()

    display_summary('../tests/evals', args.exps)