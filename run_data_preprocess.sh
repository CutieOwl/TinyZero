#!/bin/bash

set -e

if [[ -z ${DOCKER_NETWORK_DRIVER} ]]; then
	# default network is bridge
	export DOCKER_NETWORK_DRIVER="bridge"
fi

# Validate current directory for volume mounting
if [[ "$(basename "$(pwd)")" != "TinyZero" ]]; then
	echo "Error: Please run this script from the TinyZero directory."
	exit 1
fi

if [[ $# -lt 1 ]]; then
	echo "Usage: ./run_data_preprocess.sh --task_dir <path> --train_size <number> --test_size <number> --task_objective <str>"
	exit 1
fi

container_name="cybench_preprocess"  # Name of the Docker container
image_name="tinyzero/preprocess:latest"
network_name="shared_net"

# Remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
	echo "Removing existing container: ${container_name}"
	docker rm -f "${container_name}"
else 
	echo "No existing container named ${container_name} found."
fi

if ! docker image inspect "${image_name}" > /dev/null 2>&1; then
  echo "Image ${image_name} not found locally. Building it from $(pwd)/docker/preprocess/Dockerfile..."
  docker build -f $(pwd)/docker/preprocess/Dockerfile -t "${image_name}" .
else 
  echo "Image ${image_name} found locally. Using the existing image."
fi

# Create shared network if it doesn't exist
if ! docker network ls | grep -q "${network_name}"; then
	echo "Creating Docker network: ${network_name}"
	docker network create "${network_name}"
fi

get_task_dir() {
	for arg in "$@"; do
		if [[ ${arg} == "--task_dir" ]]; then
			echo "${2}"
		fi
	done
}

# if the task has a start_docker.sh, execute it
task_dir=$(get_task_dir "$@")
if [[ -f "${task_dir}/start_docker.sh" ]]; then
	echo "Executing start_docker.sh script for the task"
	"${task_dir}/start_docker.sh"
fi

# Debug information to check what is being passed
echo "Running Docker container with the following command:"

echo "docker run --name ${container_name} --network ${network_name} --rm -v $(pwd)/cybench/agent:/app/agent -v $(pwd)/examples/data_preprocess/cybench.py:/app/cybench.py --env-file=$(pwd)/cybench/.env ${image_name} python /app/cybench.py --local_dir /app/data $@"

docker run \
	--name "${container_name}" \
	-it \
	--privileged \
	--cgroupns host \
	--network "${network_name}" \
	-v "$(pwd)/cybench/agent":/app/agent:ro \
	-v "$(pwd)/examples/data_preprocess/cybench.py":/app/cybench.py:ro  \
	--env-file=$(pwd)/cybench/.env \
    "${image_name}" "$@" | tee /tmp/benchmark-latest.log
	

# Get exit status of `docker run`
exit_status=${PIPESTATUS[0]}

docker cp "${container_name}:/app/data/." "$(pwd)/data/cybench"

docker rm -f "${container_name}"

if [[ -f "${task_dir}/stop_docker.sh" ]]; then
	echo "Executing stop_docker.sh script for the task"
	"${task_dir}/stop_docker.sh"
fi

if [[ ${exit_status} -ne 0 ]]; then
	echo "Error: Task failed."
	exit 1
fi
