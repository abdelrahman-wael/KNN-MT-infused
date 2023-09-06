import { DefaultButton, TextField } from "@fluentui/react";
import React, { useState } from 'react';
import axios from 'axios';

export const Tab2 = () => {

	const [srcPath, setsrcPath] = useState('');
	const [tgtPath, settgtPath] = useState('');
	const [langDir, setlangDir] = useState('');
	// const [domainDir, setdomainDir] = useState('');
	const [isLoading, setIsLoading] = useState(false);
	const [output, setoutput] = useState(false);

	async function handleSubmit(port) {
		var data = {
			"source_data": srcPath,
			"target_data": tgtPath,
			"lang_dir": langDir,
			// "domain_id": domainDir,
		};
		setIsLoading(true);

		var config = {
			withCredentials: false,
			method: 'post',
				url: 'http://localhost:12346/index',
				headers: { 
					'Content-Type': 'application/json'
				},
			data : data
		};

		axios(config)
		.then(function (response) {
			console.log(JSON.stringify(response.data));
			setoutput(response.data["domain_id"])
			setIsLoading(false);
		})
		.catch(function (error) {
			console.log(error);
			setIsLoading(false);
		});
	}

	return (
		<div className="input">
			<TextField value={srcPath} onChange={(event) => setsrcPath(event.target.value)} className="textField" label="Source Data Path" />
			<TextField value={tgtPath} onChange={(event) => settgtPath(event.target.value)} className="textField" label="Target Data Path" />
			<TextField value={langDir} onChange={(event) => setlangDir(event.target.value)} className="textField" label="Language Direction" />
			{/* <TextField value={domainDir} onChange={(event) => setdomainDir(event.target.value)} className="textField" label="Domain Name" /> */}
			<DefaultButton className="getIndex" onClick={() => handleSubmit(5000)} disabled={isLoading} text="Create Domain" />
			<TextField value={output}  className="textField" label="Domain Identifier" />
		</div>
	);
};
