#!/bin/bash

set -e

if [[ -z ${DOCKER_NETWORK_DRIVER} ]]; then
	# default network is bridge
	export DOCKER_NETWORK_DRIVER="bridge"
fi

# Validate current directory for volume mounting
if [[ "$(basename "$(pwd)")" != "cybench" ]]; then
	echo "Error: Please run this script from the cybench directory."
	exit 1
fi

if [[ $# -lt 2 ]]; then
	echo "Usage: ./run_data_preproc.sh --task_dir <path> --train_size <number> --test_size <number>"
	exit 1
fi

container_name="cyagent"
image_name="cyagent/cyagent:latest"
network_name="shared_net"

docker rm -f "${container_name}"

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
echo "docker run --name ${container_name} --network ${network_name} --rm -v $(pwd)/agent:/app/agent -v $(pwd)/data_preproc_cybench.py:/app/data_preproc_cybench.py --env-file=.env ${image_name} python /app/data_preproc_cybench.py  --local_dir /app/data $@"

docker run \
	--name "${container_name}" \
	-it \
	--privileged \
	--network "${network_name}" \
	--cgroupns host \
	-v "$(pwd)/agent":/app/agent:ro \
	-v "$(pwd)/data_preproc_cybench.py":/app/data_preproc_cybench.py:ro \
	--env-file=.env \
	"${image_name}" "$@" 
	# --entrypoint "/usr/local/bin/dockerd-entrypoint.sh" "${image_name}" "python3" "data_preproc_cybench.py --local_dir /app/data $@"

# Get exit status of `docker run`
exit_status=${PIPESTATUS[0]}

docker cp "${container_name}:/app/data/." "${parent_dir}/data/cybench_docker"

docker rm -f "${container_name}"

if [[ -f "${task_dir}/stop_docker.sh" ]]; then
	echo "Executing stop_docker.sh script for the task"
	"${task_dir}/stop_docker.sh"
fi

if [[ ${exit_status} -ne 0 ]]; then
	echo "Error: Task failed."
	exit 1
fi
