# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import os


def specialize_source(source, specialize_for, search_in_folders=[]):

    assert specialize_for in ["cpu_serial", "cpu_openmp", "opencl", "cuda"]

    source_lines = source.splitlines()

    lines = []
    for ll in source_lines:
        if "//include_file" in ll:
            assert " for_context " in ll
            fname = (
                ll.split("//include_file")[-1].split("for_context")[0].strip()
            )
            temp_contexts = ll.split("for_context")[-1].split()
            temp_contexts = [ss.strip() for ss in temp_contexts]
            if specialize_for in temp_contexts:
                for fold in ["./"] + search_in_folders:
                    fpath = fold + "/" + fname
                    if os.path.isfile(fpath):
                        break
                else:
                    raise IOError(f"File {fname} not found")
                with open(fpath, "r") as fid:
                    flines = fid.readlines()
                lines.append("\n//from file: " + fname + "\n")
                for fll in flines:
                    lines.append(fll.rstrip())
                lines.append("\n//end file: " + fname + "\n")
        else:
            lines += ll.split("\n")

    indent = False
    new_lines = []
    inside_vect_block = False
    for ii, ll in enumerate(lines):
        if "//vectorize_over" in ll:
            if inside_vect_block:
                raise ValueError(f"Line {ii}: Previous vect block not closed!")
            inside_vect_block = True
            varname, limname = ll.split("//vectorize_over")[-1].split()
            if specialize_for.startswith("cpu"):
                new_lines.append(
                    f"for (int {varname}=0; {varname}<{limname}; {varname}++)"
                    + "{ //autovectorized\n"
                )
            elif specialize_for == "opencl":
                new_lines.append(f"int {varname}; //autovectorized\n")
                new_lines.append(
                    f"{varname}=get_global_id(0); //autovectorized\n"
                )

            elif specialize_for == "cuda":
                new_lines.append(f"int {varname}; //autovectorized\n")
                new_lines.append(
                    f"{varname}=blockDim.x * blockIdx.x + threadIdx.x;"
                    "//autovectorized\n"
                    f"if ({varname}<{limname})" + "{"
                )
        elif "//end_vectorize" in ll:
            if specialize_for.startswith("cpu"):
                new_lines.append("}//end autovectorized\n")
            elif specialize_for == "opencl":
                new_lines.append("//end autovectorized\n")
            elif specialize_for == "cuda":
                new_lines.append("}//end autovectorized\n")

            inside_vect_block = False
        else:
            if "//only_for_context" in ll:
                temp_contexts = ll.split("//only_for_context")[-1].split()
                temp_contexts = [ss.strip() for ss in temp_contexts]
                if specialize_for not in temp_contexts:
                    ll = "//" + ll
            if indent and inside_vect_block:
                new_lines.append("    " + ll)
            else:
                new_lines.append(ll)

    newfilecontent = "\n".join(new_lines)
    newfilecontent = newfilecontent.replace(
        "/*gpukern*/",
        {
            "cpu_serial": " ",
            "cpu_openmp": " ",
            "opencl": " __kernel ",
            "cuda": "__global__",
        }[specialize_for],
    )
    newfilecontent = newfilecontent.replace(
        "/*gpufun*/",
        {
            "cpu_serial": " static inline",
            "cpu_openmp": " static inline",
            "opencl": " ",
            "cuda": " __device__ ",
        }[specialize_for],
    )
    newfilecontent = newfilecontent.replace(
        "/*gpuglmem*/",
        {
            "cpu_serial": " ",
            "cpu_openmp": " ",
            "opencl": " __global ",
            "cuda": " ",
        }[specialize_for],
    )

    if os.name == "nt":  # windows
        restrict_qualifier = " "
    else:  # other os
        restrict_qualifier = " restrict "
    newfilecontent = newfilecontent.replace(
        "/*restrict*/",
        {
            "cpu_serial": restrict_qualifier,
            "cpu_openmp": restrict_qualifier,
            "opencl": "",
            "cuda": "",
        }[specialize_for],
    )

    return newfilecontent
