import firecrest as fc
machine = "daint"
client_id = "firecrest-lisergey-shell"
client_secret = ""
token_uri = "https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"
authorization = fc.ClientCredentialsAuth(client_id, client_secret, token_uri)
client = fc.Firecrest(firecrest_url="https://firecrest.cscs.ch",
                      authorization=authorization)
job = client.submit(machine, "examples/script.sh", account="d124")
out = job["job_file_out"]
err = job["job_file_out"]
id = job["jobid"]
status = client.poll(machine, [id])
client.simple_download(machine, out, "out")
