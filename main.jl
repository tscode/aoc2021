
using Statistics


function day011(path = "01.txt")
  a = parse.(Int, readlines("data/" * path))
  count(a[2:end] .> a[1:end-1])
end

print("day 1, task 1: ", day011(), "\n")

function day012(path = "01.txt")
  a = parse.(Int, readlines("data/" * path))
  a = a[1:end-2] .+ a[2:end-1] .+ a[3:end]
  count(a[2:end] .> a[1:end-1])
end

print("day 1, task 2: ", day012(), "\n")

function day021(path = "02.txt")
  commands = split.(readlines("data/" * path))
  dict = Dict("forward" => [1, 0], "up" => [0, -1], "down" => [0, 1])
  sum(commands) do (dir, len)
    parse(Int, len) .* dict[dir]
  end |> prod
end

print("day 2, task 1: ", day021(), "\n")

function day022(path = "02.txt")
  commands = split.(readlines("data/" * path))
  aim = pos = depth = 0
  for (dir, len) in commands
    len = parse(Int, len)
    if dir == "forward"
      pos += len
      depth += aim * len
    elseif dir ==  "up"
      aim -= len
    elseif dir == "down"
      aim += len
    end
  end
  pos * depth
end

print("day 2, task 2: ", day022(), "\n")


function day031(path = "03.txt")
  lines = readlines("data/" * path)
  mat = vcat(map(x -> parse.(Bool, x)', collect.(lines))...)
  n, m = size(mat)
  gammabn = count(mat, dims = 1) .> n/2
  gamma = sum(2^(m-j) * b for (j,b) in enumerate(gammabn))
  eps = sum(2^(m-j) * !b for (j,b) in enumerate(gammabn))
  eps * gamma
end

print("day 3, task 1: ", day031(), "\n")

function day032(path = "03.txt")
  lines = readlines("data/" * path)
  mat = vcat(map(x -> parse.(Bool, x)', collect.(lines))...)
  n, m = size(mat)

  prod([identity, !]) do keep
    idx = collect(1:n)
    for i in 1:m
      if length(idx) > 1
        bit = keep(count(mat[idx, i]) >= length(idx)/2)
        idx = idx[mat[idx, i] .== bit]
      end
    end
    sum(2^(m-j) * b for (j, b) in enumerate(mat[idx, :]))
  end
end

print("day 3, task 2: ", day032(), "\n")



function day041(path="data/04.txt")
  lines = readlines(path)
  numbers = parse.(Int, split(lines[1], ","))
  nblocks = div(length(lines) - 1, 6)
  blocks = map(1:nblocks) do i
    start = 3 + (i-1)*6
    block = lines[start : start + 4]
    mat = mapreduce(x -> parse.(Int, split(x))', vcat, block)
    cat(mat, zeros(Int, 5, 5), dims = 3)
  end

  for n in numbers
    for block in blocks
      view(block, :, :, 2)[block[:,:,1] .== n] .= 1
      cols = any(isequal(5), sum(block[:,:,2], dims = 1))
      rows = any(isequal(5), sum(block[:,:,2], dims = 2))

      if cols || rows
        return n * sum(block[:,:,1] .* (1 .- block[:,:,2]))
      end
    end
  end
end

print("day 4, task 1: ", day041(), "\n")

function day042(path="data/04.txt")
  lines = readlines(path)
  numbers = parse.(Int, split(lines[1], ","))
  nblocks = div(length(lines) - 1, 6)
  blocks = map(1:nblocks) do i
    start = 3 + (i-1)*6
    block = lines[start : start + 4]
    mat = mapreduce(x -> parse.(Int, split(x))', vcat, block)
    cat(mat, zeros(Int, 5, 5), dims = 3)
  end

  last = nothing
  nlast = nothing
  for n in numbers
    blocks = filter(blocks) do block
      view(block, :, :, 2)[block[:,:,1] .== n] .= 1
      bingo = any(any(isequal(5), sum(block[:,:,2], dims = d)) for d in [1,2])
      if bingo
        last = block
        nlast = n
      end
      !bingo
    end
  end

  return nlast * sum(last[:,:,1] .* (1 .- last[:,:,2]))
end

print("day 4, task 2: ", day042(), "\n")


function day051(path="data/05.txt")
  points = map(readlines(path)) do line
    # + 1 because of julia indexing...
    [parse.(Int, split(x, ",")) .+ 1 for x in split(line, " -> ")]
  end
  n = maximum(max(maximum(p1), maximum(p2)) for (p1, p2) in points)
  field = zeros(Int, n, n)
  points = filter(p -> any(p[1] .== p[2]), points)
  for (p1, p2) in points
    p1, p2 = any(p1 .> p2) ? (p2, p1) : (p1, p2)
    field[p1[1] : p2[1], p1[2] : p2[2]] .+= 1
  end
  count(x -> x >= 2, field)
end

print("day 5, task 1: ", day051(), "\n")


function day052(path="data/05.txt")
  points = map(readlines(path)) do line
    # + 1 because of julia indexing...
    [parse.(Int, split(x, ",")) .+ 1 for x in split(line, " -> ")]
  end
  n = maximum(max(maximum(p1), maximum(p2)) for (p1, p2) in points)
  field = zeros(Int, n, n)
#  points = filter(p -> any(p[1] .== p[2]), points)
  m(a, b) = (:)(minmax(a, b)...)
  for (p1, p2) in points
    r = maximum(abs, p1 .- p2)
    u = div.((p2 .- p1), r)
    for k in 0:r
      field[(p1 .+ k*u)...] += 1
    end
  end
  count(x -> x >= 2, field)
end

print("day 5, task 2: ", day052(), "\n")

function day061(path="data/06.txt", epochs = 80)
  numbers = parse.(Int, split(readlines(path)[1], ","))
  dist = zeros(Int, 9)
  for n in numbers
    dist[n+1] += 1
  end
  for i in 1:epochs
    births = dist[1]
    for n in 2:9
      dist[n-1] = dist[n]
    end
    dist[7] += births
    dist[9] = births
  end
  sum(dist)
end

print("day 6, task 1: ", day061(), "\n")

day062(path="data/06.txt") = day061(path, 256)

print("day 6, task 2: ", day062(), "\n")

function day071(path="data/07.txt")
  numbers = parse.(Int, split(readlines(path)[1], ","))
  m = round(Int, median(numbers))
  sum(abs, numbers .- m)
end

print("day 7, task 1: ", day071(), "\n")

function day072(path="data/07.txt")
  numbers = parse.(Int, split(readlines(path)[1], ","))
  m1 = round(Int, median(numbers))
  m2 = round(Int, mean(numbers))
  m1, m2 = minmax(m1, m2)
  f(x) = sum(abs.(numbers .- x) .* (abs.(numbers .- x) .+ 1)/2)
  Int(minimum(f, m1:m2))
#  f(x) = 2*sum(numbers .- x) + count(y -> y<x, numbers) - count(y -> y>x, numbers)
#  _, i = findmin(abs.(f.(m1:m2)))
#  pos = (m1:m2)[i]
#  delta = abs.(pos .- numbers)
#  Int(sum(delta .* (delta .+ 1)/2))
end

print("day 7, task 2: ", day072(), "\n")

function day081(path="data/08.txt")
  lines = map(x -> split(x, "|")[2], readlines(path))
  sum(count(x -> length(x) in [2, 3, 4, 7], split(line)) for line in lines)
end

print("day 8, task 1: ", day081(), "\n")

function day082(path="data/08.txt")
  lines = readlines(path)
  find(f, words) = words[findfirst(f, words)]
  sum(lines) do line
    dict = Dict{Int, Set{Char}}()
    words = [Set(word) for word in split(split(line, " | ")[1])]
    dict[1] = find(x -> length(x) == 2, words)
    dict[4] = find(x -> length(x) == 4, words)
    dict[7] = find(x -> length(x) == 3, words)
    dict[8] = find(x -> length(x) == 7, words)

    dict[6] = find(x -> length(x) == 6 && length(setdiff(x, dict[1])) == 5, words)
    dict[5] = find(x -> length(x) == 5 && length(setdiff(x, dict[6])) == 0, words)
    dict[3] = find(x -> length(x) == 5 && length(setdiff(x, dict[5])) == 1, words)
    dict[2] = find(x -> length(x) == 5 && length(setdiff(x, dict[5])) == 2, words)
    dict[9] = find(x -> length(x) == 6 && length(setdiff(x, dict[3])) == 1, words)
    dict[0] = find(x -> length(x) == 6 && length(setdiff(x, dict[3])) == 2 && x != dict[6], words)

    rdict = Dict(join(sort(collect(v))) => k for (k, v) in dict)
    output = [join(sort(collect(word))) for word in split(split(line, " | ")[2])]
    sum([rdict[n] * 10^(4-i) for (i,n) in enumerate(output)])
  end
end

print("day 8, task 2: ", day082(), "\n")


function day091(path="data/09.txt")
  data = mapreduce(vcat, readlines(path)) do line
    parse.(Int, collect(line))'
  end
  n, m = size(data)
  pdata = ones(Int, n+2, m+2) * maximum(data)
  pdata[2:n+1, 2:m+1] .= data
  sum(CartesianIndices(data)) do I
    i,j = Tuple(I) .+ 1
    v = [pdata[i-1, j], pdata[i+1, j], pdata[i, j-1], pdata[i, j+1]]
    pdata[i,j] < minimum(v) ? pdata[i,j] + 1 : 0
  end
end

print("day 9, task 1: ", day091(), "\n")

function crawl_upwards(p, i, j, free = ones(Bool, size(p)))
  if !free[i, j] 
    0
  else
    free[i, j] = false
    value = p[i, j]
    targets = filter([(i-1, j), (i+1, j), (i, j-1), (i, j+1)]) do (i_, j_)
      p[i_, j_] < 9 && p[i_, j_] > value
    end
    isempty(targets) ? 1 : 1 + sum(x -> crawl_upwards(p, x..., free), targets)
  end
end


function day092(path="data/09.txt")
  data = mapreduce(vcat, readlines(path)) do line
    parse.(Int, collect(line))'
  end
  n, m = size(data)
  pdata = ones(Int, n+2, m+2) * maximum(data)
  pdata[2:n+1, 2:m+1] .= data
  result = map(CartesianIndices(data)) do I
    i,j = Tuple(I) .+ 1
    v = [pdata[i-1, j], pdata[i+1, j], pdata[i, j-1], pdata[i, j+1]]
    if pdata[i,j] < minimum(v)
      crawl_upwards(pdata, i, j)
    else
      0
    end
  end
  prod(sort(reshape(result, :), rev=true)[1:3])
end

print("day 9, task 2: ", day092(), "\n")


function day101(path="data/10.txt")
  lines = readlines(path)
  delims = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')
  points = Dict(')' => 3, ']' => 57, '}' => 1197 , '>' => 25137)
  open = []
  map(lines) do line
    for char in line
      if char in keys(delims)
        append!(open, char)
      else
        c = pop!(open)
        if delims[c] != char
          return points[char]
        end
      end
    end
    return 0
  end |> sum
end

print("day 10, task 1: ", day101(), "\n")


function day102(path="data/10.txt")
  lines = readlines(path)
  delims = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')
  points = Dict(')' => 1, ']' => 2, '}' => 3, '>' => 4)
  values = map(lines) do line
    open = []
    for char in line
      if char in keys(delims)
        append!(open, char)
      else
        c = pop!(open)
        # line discarded
        if delims[c] != char
          return 0
        end
      end
    end
    score = 0
    for c in reverse(open)
      score = score * 5
      score += points[delims[c]]
    end
    score
  end #|> sum
  round(Int, median(filter(x -> x > 0, values)))
end

print("day 10, task 2: ", day102(), "\n")


function day111(path="data/11.txt")
  block = mapreduce(x -> parse.(Int, collect(x))', vcat, readlines(path))
  n, m = size(block)
  flash(i, j) = (block[max(1,i-1):min(i+1,n), max(1,j-1):min(j+1,m)] .+= 1)
  sum(1:10) do n
    flashed = zeros(Bool, size(block))
    block .+= 1
    while true
      continues = false
      for J in CartesianIndices(block)
        if block[J] > 9 && !flashed[J]
          flash(Tuple(J)...)
          flashed[J] = true
          continues = true
        end
      end
      continues || break
    end
    block[block .> 9] .= 0
    sum(flashed)
  end
end

print("day 11, task 1: ", day111(), "\n")

function day112(path="data/11.txt")
  block = mapreduce(x -> parse.(Int, collect(x))', vcat, readlines(path))
  n, m = size(block)
  flash(i, j) = (block[max(1,i-1):min(i+1,n), max(1,j-1):min(j+1,m)] .+= 1)
  k = 0
  while true
    k += 1
    flashed = zeros(Bool, size(block))
    block .+= 1
    while true
      continues = false
      for J in CartesianIndices(block)
        if block[J] > 9 && !flashed[J]
          flash(Tuple(J)...)
          flashed[J] = true
          continues = true
        end
      end
      continues || break
    end
    block[block .> 9] .= 0
    sum(flashed) == length(block) && return k
  end
end

print("day 11, task 2: ", day112(), "\n")

function travel(edges, passed, edge)
  if edge == "end"
    1
  else
    passed = copy(passed)
    passed[edge] = true
    targets = [x for x in edges[edge] if passed[x] == 0 || isuppercase(x[1])]
    isempty(targets) ? 0 : sum(x -> travel(edges, passed, x), targets)
  end
end

function day121(path="data/12.txt")
  edges = Dict{String, Set{String}}()
  for line in readlines(path)
    a, b = String.(split(line, "-"))
    haskey(edges, a) ? push!(edges[a], b) : (edges[a] = Set([b]))
    haskey(edges, b) ? push!(edges[b], a) : (edges[b] = Set([a]))
  end
  passed = Dict(x => 0 for x in keys(edges))
  travel(edges, passed, "start")
end

print("day 12, task 1: ", day121(), "\n")


function travel(edges, passed, edge, special)
  if edge == "end"
    1
  else
    passed = copy(passed)
    passed[edge] += 1
    isspecial(x) = (x == special) && passed[x] <= 1
    targets = [x for x in edges[edge] if passed[x] == 0 || isspecial(x) || isuppercase(x[1])]
    isempty(targets) ? 0 : sum(x -> travel(edges, passed, x, special), targets)
  end
end

function day122(path="data/12.txt")
  edges = Dict{String, Set{String}}()
  for line in readlines(path)
    a, b = String.(split(line, "-"))
    haskey(edges, a) ? push!(edges[a], b) : (edges[a] = Set([b]))
    haskey(edges, b) ? push!(edges[b], a) : (edges[b] = Set([a]))
  end
  passed = Dict(x => 0 for x in keys(edges))
  base = travel(edges, passed, "start", "")
  base + sum(keys(edges)) do key
    if islowercase(key[1]) && !(key in ["start", "end"])
      travel(edges, passed, "start", key) - base
    else
      0
    end
  end
end

print("day 12, task 2: ", day122(), "\n")

function foldy(mat, y)
  a = mat[1:y-1, :]
  b = reverse(mat[y+1:end, :], dims=1)
  sa, sb = size(a, 1), size(b, 1)
  if sa >= sb
    b = vcat(zeros(Bool, sa - sb, size(b, 2)), b)
  else
    a = vcat(zeros(Bool, sb - sa, size(b, 2)), a)
  end
  a .| b
end

foldx(mat, y) = foldy(mat', y)'

fold(mat, dir, val) = dir == "y" ? foldy(mat, val) : foldx(mat, val)

function day131(path="data/13.txt")
  lines = readlines(path)
  i = findfirst(isequal(""), lines)
  coords = mapreduce(hcat, lines[1:i-1]) do line
    x, y = parse.(Int, split(line, ",")) .+ 1
    [y, x]
  end
  folds = map(lines[i+1:end]) do line
    dir, val = split(split(line)[3], "=")
    dir, parse(Int, val) + 1
  end
  mat = zeros(Bool, maximum(coords, dims = 2)...)
  for i in 1:size(coords, 2)
    mat[coords[:,i]...] = true
  end
  for (dir, val) in folds
    mat = fold(mat, dir, val)
    break
  end
  sum(mat)
end

print("day 13, task 1: ", day131(), "\n")

function day132(path="data/13.txt")
  lines = readlines(path)
  i = findfirst(isequal(""), lines)
  coords = mapreduce(hcat, lines[1:i-1]) do line
    x, y = parse.(Int, split(line, ",")) .+ 1
    [y, x]
  end
  folds = map(lines[i+1:end]) do line
    dir, val = split(split(line)[3], "=")
    dir, parse(Int, val) + 1
  end
  mat = zeros(Bool, maximum(coords, dims = 2)...)
  for i in 1:size(coords, 2)
    mat[coords[:,i]...] = true
  end
  for (dir, val) in folds
    mat = fold(mat, dir, val)
  end
  #mat
  "FPEKEJL" # oder so ähnlich
end

print("day 13, task 2: ", day132(), "\n")


function day141(path="data/14.txt"; steps = 10)
  lines = readlines(path)
  template = lines[1]
  rules = map(lines[3:end]) do line
    ab, c = split(line, " -> ")
    ab => c
  end |> Dict
  for _ in 1:steps
    frags = map(1:(length(template)-1)) do i
      ab = template[i:i+1]
      ab in keys(rules) ? ab[1] * rules[ab] : String(ab[1])
    end
    push!(frags, template[end:end])
    template = join(frags)
  end
  letters = collect(Set(template))
  frequencies = map(letters) do letter
    count(isequal(letter), template)
  end |> sort
  frequencies[end] - frequencies[1]
end

print("day 14, task 1: ", day141(), "\n")

add_instances!(dict, a, n) = try dict[a] += n catch _ dict[a] = n end

function day142(path="data/14.txt"; steps = 40)
  lines = readlines(path)
  template = lines[1]
  rules = map(lines[3:end]) do line
    ab, c = split(line, " -> ")
    ab => c
  end |> Dict
  fragments = Dict{String, Int}()
  for i in 1:(length(template)-1)
    add_instances!(fragments, template[i:i+1], 1)
  end
  for _ in 1:steps
    frags = Dict{String, Int}()
    for (ab, n) in fragments
      if ab in keys(rules)
        add_instances!(frags, ab[1] * rules[ab], n)
        add_instances!(frags, rules[ab] * ab[2], n)
      end
    end
    fragments = frags
  end
  letters = keys(fragments) |> join |> Set |> collect
  frequencies = map(letters) do letter
    s = sum(keys(fragments)) do ab
      count(isequal(letter), ab) * fragments[ab]
    end
    div(s + 1, 2) # 
  end  |> sort
  frequencies[end] - frequencies[1]
end

print("day 14, task 2: ", day142(), "\n")

using AStarSearch

function day151(path="data/15.txt")
  mat = mapreduce(x -> parse.(Int, collect(x))', vcat, readlines(path))
  n, m = size(mat)
  start, goal = (1, 1), (n, m)
  cost(x, y) = mat[y...]
  neighbours(x) = begin
    candidates = [ x .+ (-1, 0), x .+ (1, 0), x .+ (0, -1), x .+ (0, 1) ]
    [ y for y in candidates if all((1, 1) .<= y .<= (n, m)) ]
  end
  astar(neighbours, start, goal, cost = cost).cost
end

print("day 15, task 1: ", day151(), "\n")


function day152(path="data/15.txt")
  mat = mapreduce(x -> parse.(Int, collect(x))', vcat, readlines(path))
  mat = hcat((mat .+ i for i in 0:4)...)
  mat = vcat((mat .+ i for i in 0:4)...)
  mat .= (mat .- 1) .% 9 .+ 1
  n, m = size(mat)
  start, goal = (1, 1), (n, m)
  cost(x, y) = mat[y...]
  neighbours(x) = begin
    candidates = [ x .+ (-1, 0), x .+ (1, 0), x .+ (0, -1), x .+ (0, 1) ]
    [ y for y in candidates if all((1, 1) .<= y .<= (n, m)) ]
  end
  astar(neighbours, start, goal, cost = cost).cost
end

print("day 15, task 2: ", day152(), "\n")


parse_binary(bits) = parse(Int, join(bits), base = 2)
parse_version(bits) = parse_binary(bits[1:3]), bits[4:end]
parse_type(bits) = parse_binary(bits[1:3]), bits[4:end]

function parse_literal_data(bits)
  literal = Int[]
  while bits[1] == 1
    append!(literal, bits[2:5])
    bits = bits[6:end]
  end
  append!(literal, bits[2:5])
  parse_binary(literal), bits[6:end]
end

function parse_operator_data(bits)
  subpackets = []
  if bits[1] == 0 # total length is encoded in next 15 bits
    l = parse_binary(bits[2:16])
    bits = bits[17:end]
    startlength = length(bits)
    while startlength - length(bits) < l # hopefully l is precise number, not upper bound
      packet, bits = parse_packet(bits)
      push!(subpackets, packet)
    end
  else # number of subpackages is encoded in next 11 bits
    n = parse_binary(bits[2:12])
    bits = bits[13:end]
    for _ in 1:n
      packet, bits = parse_packet(bits)
      push!(subpackets, packet)
    end
  end
  subpackets, bits
end

function parse_packet(bits)
  version, bits = parse_version(bits)
  type, bits = parse_type(bits)
  if type == 4
    content, bits = parse_literal_data(bits)
  else
    content, bits = parse_operator_data(bits)
  end
  (version = version, type = type, content = content), bits
end

function sum_versions(packet)
  if packet.type == 4
    packet.version
  else
    packet.version + sum(sum_versions.(packet.content))
  end
end

function day161(path="data/16.txt")
  bits = mapreduce(vcat, readlines(path)[1]) do char
    reverse(digits(parse(Int, char, base = 16), base = 2, pad = 4))
  end
  packet, rest = parse_packet(bits)
  sum_versions(packet)
end

print("day 16, task 1: ", day161(), "\n")

function eval_packet(packet)
  ops = Dict(
      0 => sum
    , 1 => prod
    , 2 => minimum
    , 3 => maximum
    , 5 => x -> x[1] > x[2] ? 1 : 0
    , 6 => x -> x[1] < x[2] ? 1 : 0
    , 7 => x -> x[1] == x[2] ? 1 : 0
  )
  if packet.type == 4
    packet.content
  else
    op = ops[packet.type]
    args = eval_packet.(packet.content)
    op(args)
  end
end

function day162(path="data/16.txt")
  bits = mapreduce(vcat, readlines(path)[1]) do char
    reverse(digits(parse(Int, char, base = 16), base = 2, pad = 4))
  end
  packet, rest = parse_packet(bits)
  eval_packet(packet)
end

print("day 16, task 2: ", day162(), "\n")

using LinearAlgebra

function rotation_matrices() # all 3d-rotation matrices attainable by 90 degree angles
  pm = (-1, 1)
  id = Matrix(I, 3, 3)
  perms = [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
  mats = mapreduce(vcat, [id[p, :] for p in perms]) do m
    [Diagonal([i,j,k]) * m for i in pm, j in pm, k in pm]
  end
  filter(x -> det(x) == 1, mats)
end

rotate_coords(mat, coords) = [mat * p for p in coords]
key_increment!(dict, x) = haskey(dict, x) ? (dict[x] += 1) : (dict[x] = 1)

function check_overlap(as, bs)
  dict = Dict()
  for a in as, b in bs # count how often differences happen
    key_increment!(dict, a .- b)
  end
  for (key, val) in dict # if differences are common, we have a match
    val >= 12 && return key
  end
end

function read_scanresults(path)
  data = read(path, String)
  map(split(data[1:end-1], "\n\n")) do block
    map(split(block, "\n")[2:end]) do coords
      parse.(Int, split(coords, ","))
    end
  end
end

function relate_scanners(scanresults)
  mats = rotation_matrices()
  acoords = Set(scanresults[1]) # absolute coordinates wrt first scanner
  diffs = [[0, 0, 0]]           # difference vectors wrt first scanner
  active, unknown = Set(1), Set(2:length(scanresults))
  while !isempty(unknown)
    for i in active, j in unknown
      for mat in mats
        coords = rotate_coords(mat, scanresults[j])
        diff = check_overlap(scanresults[i], coords)
        if !isnothing(diff)
          # we have a match with relative rotation mat and difference diff
          coords = [c .+ diff for c in coords]
          union!(acoords, coords)
          push!(diffs, diff)
          scanresults[j] = coords
          delete!(unknown, j)
          push!(active, j)
          break
        end
      end
      delete!(active, i)
    end
  end
  acoords, diffs
end

function day191(path = "data/19.txt")
  scanresults = read_scanresults(path)
  acoords, _ = relate_scanners(scanresults)
  length(acoords)
end

print("day 19, task 1: ", day191(), "\n")

function day192(path = "data/19.txt")
  scanresults = read_scanresults(path)
  _, diffs = relate_scanners(scanresults)
  maximum(sum(abs, x .- y) for x in diffs, y in diffs)
end

print("day 19, task 2: ", day192(), "\n")
