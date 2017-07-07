#' Update list function used in allUncovPath function
#' @param path The path to update
#' @param set Set of vertices to update path with
#' @param path old.list The old list of paths
#' @return List containing all updated paths

updateList_m <- function (path, set, old.list)
{
  c(old.list, lapply(set, function(s) c(path, s)))
}
