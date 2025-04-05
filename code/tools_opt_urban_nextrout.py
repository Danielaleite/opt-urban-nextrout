import sys
import numpy as np
from itertools import combinations
import os

# required scripts from Nextrout

import sys

sys.path.append("./Nextrout/nextrout_core/")

import dmk_cont
import filtering
import pre_extraction
from scipy.spatial.distance import cdist


def set_forcing(xplus, xminus):

    fplus = 1 / len(xplus) * np.ones(len(xplus))
    fminus = 1 / len(xminus) * np.ones(len(xminus))

    Nplus = len(xplus)
    Nminus = len(xminus)

    extra_info = {
        "Nplus": Nplus,
        "Nminus": Nminus,
        "fplus": fplus,
        "fminus": fminus,
        "xplus": xplus,
        "xminus": xminus,
    }

    return extra_info


def run_nextrout(xplus, xminus, beta_c, niter):
    extra_info = set_forcing(xplus, xminus)

    ndiv = 32
    forcing_flag = "dirac"

    (
        grid,
        subgrid,
        points,
        vertices,
        coord,
        topol,
        element_attributes,
    ) = dmk_cont.grid_gen(ndiv)

    (forcing, triang_source_indices, triang_sink_indices,) = dmk_cont.forcing_generator(
        forcing_flag, grid, coord, topol, extra_info=extra_info
    )

    tdpot, timefun = dmk_cont.dmk_cont(forcing, beta_c, ndiv, extra_info, niter)

    tdens_weights = tdpot.tdens
    tdens_weights = tdens_weights / max(tdens_weights)

    Gpe = pre_extraction.pre_extr(
        coord,
        topol,
        tdens_weights,
        triang_source_indices,
        triang_sink_indices,
        min_=1e-5,
        graph_type="1",
    )

    return (tdpot, timefun, Gpe)


source_sink_path = "../data/input/"

########## Find set (x,y) for source/sink coordinates


def find_xy_coords(city, t_mode, centrality, target, component=0):

    with open(
        # "../../data/input/source_sink_files/dirac-sources/"
        source_sink_path
        + str(city)
        + "-sources-sinks/"
        + str(target)
        + "_"
        + str(component)
        + "_"
        + str(city)
        + "_"
        + str(t_mode)
        + "_"
        + str(centrality)
        + ".dat",
        "rb",
    ) as file:
        sources_file = [i.strip().split() for i in file.readlines()]
        file.close()

    source_list = [(float(x[0]), float(x[1])) for x in sources_file]

    return source_list


########## This functions are used for the discrete case #######


def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]


def find_sources(graph, city, t_mode, centrality, component=0):

    with open(
        # "../../data/input/source_sink_files/dirac-sources/"
        source_sink_path
        + str(city)
        + "-sources-sinks/"
        + "source_"
        + str(component)
        + "_"
        + str(city)
        + "_"
        + str(t_mode)
        # + "_"
        # + str(centrality)
        + ".dat",
        "rb",
    ) as file:
        sources_file = [i.strip().split() for i in file.readlines()]
        file.close()

    source_list = [(float(x[0]), float(x[1])) for x in sources_file]

    node_x = []
    node_y = []
    n = []
    """
    if nx.is_tree(graph) == True:
        for node in graph.nodes():
            if graph.degree(node) <= 2:
                x, y = graph.nodes[node]["pos"]
                node_x.append(x)
                node_y.append(y)
                n.append(node)
    """

    try:
        for node in graph.nodes():

            x, y = graph.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
            n.append(node)
    except:
        pass

    xy_list = list(zip(node_x, node_y))

    xy_list = list(zip(node_x, node_y))
    xy_list = np.asarray(xy_list)
    source = np.asarray(source_list)

    xynode_list = list(zip(n, xy_list))

    sourcelist = []
    for i, values in enumerate(source):
        sourcelist.append((i, closest_node(values, xy_list)))

    final_sources = []
    for i in sourcelist:
        for j in xynode_list:
            if (i[1] == j[1]).all():
                final_sources.append(int(j[0]))

    finals = list(dict.fromkeys(final_sources))

    return finals


def find_sinks(graph, city, t_mode, centrality, component=0):
    with open(
        # "../../data/input/source_sink_files/dirac-sources/"
        source_sink_path
        + str(city)
        + "-sources-sinks/"
        + "sink_"
        + str(component)
        + "_"
        + str(city)
        + "_"
        + str(t_mode)
        + "_"
        + str(centrality)
        + ".dat",
        "rb",
    ) as file:
        sink_file = [i.strip().split() for i in file.readlines()]
        file.close()

    sink_list = [(float(x[0]), float(x[1])) for x in sink_file]

    node_x = []
    node_y = []
    n = []
    for node in graph.nodes():
        x, y = graph.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        n.append(node)

    xy_list = list(zip(node_x, node_y))

    xy_list = list(zip(node_x, node_y))
    xy_list = np.asarray(xy_list)
    sink = np.asarray(sink_list)

    xynode_list = list(zip(n, xy_list))

    sinklist = []
    for i, values in enumerate(sink):
        sinklist.append((i, closest_node(values, xy_list)))

    final_sinks = []
    for i in sinklist:
        for j in xynode_list:
            if (i[1] == j[1]).all():
                final_sinks.append(int(j[0]))

    # print("Looking for sinks: ", final_sinks)

    finals = list(dict.fromkeys(final_sinks))

    return finals


def graph_filtering(Graph, sources, sinks, beta_d):

    Gf, weights, colors, inputs_discr = filtering.filtering(
        Graph,
        sources,
        sinks,
        beta_d=beta_d,
        tdens0=2,  # 2 means not unitary (i.e., taken from Gpe)
        threshold=1e-4,
        weight_flag="length",
        stopping_threshold_f=1e-15,
    )

    return Gf, weights, colors, inputs_discr
