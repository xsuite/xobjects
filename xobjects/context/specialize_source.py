def specialize_source(source, specialize_for):

    assert specialize_for in ["cpu_serial", "cpu_openmp", "opencl", "cuda"]

    lines = source.splitlines()

    indent = True
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
                )
        elif "//end_vectorize" in ll:
            if specialize_for.startswith("cpu"):
                new_lines.append("}//end autovectorized\n")
            elif specialize_for == "opencl":
                new_lines.append("//end autovectorized\n")
            elif specialize_for == "cuda":
                new_lines.append("//end autovectorized\n")

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
        {"cpu_serial": " ", "cpu_openmp": " ",
            "opencl": " __kernel ", "cuda": "__global__"}[
            specialize_for
        ],
    )
    newfilecontent = newfilecontent.replace(
        "/*gpuglmem*/",
        {"cpu_serial": " ", "cpu_openmp": " ",
         "opencl": " __global ", "cuda": " "}[specialize_for],
    )

    return newfilecontent
