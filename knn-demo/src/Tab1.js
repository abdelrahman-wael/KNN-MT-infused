import { ActionButton, DefaultButton, TextField } from "@fluentui/react";
import React, { useState } from 'react';
import axios from 'axios';
import { Spinner } from "@fluentui/react-spinner";


export const Tab1 = () => {
	const [inputValue, setInputValue] = useState('');
	const [domainValue, setDomainValue] = useState('');
	const [outputValue, setOutputValue] = useState('');
	const [noKnnValue, setnoKnnValue] = useState('');
	const [isLoading, setIsLoading] = useState(false);
	const [languageDir, setlanguageDir] = useState("");

	async function handleSubmit(use_knn) {
		console.log(inputValue)
		var data = {
			"document": inputValue.split("\n"),
			"lang_dir": languageDir,
			"use_knn": use_knn,
			"batch_size":32,
			"domain_id": [domainValue, "asdhg"]
		};
		setIsLoading(true);

		var config = {
			withCredentials: false,
			method: 'post',
				url: 'http://localhost:12346/translate',
				headers: { 
					'Content-Type': 'application/json'
				},
			data : data
		};

		axios(config)
		.then(function (response) {
			console.log(JSON.stringify(response.data));
			if (use_knn) {
				setOutputValue(response.data.join("\n"))
			}
			else {
				setnoKnnValue(response.data.join("\n"))
			}
			setIsLoading(false);
		})
		.catch(function (error) {
			console.log(error);
			setIsLoading(false);
		});
	}

	return (
		<div className="page">
		 	        {/* <Spinner appearance="primary" label="Primary Spinner" /> */}
			<div className="tableContainer">
				<TextField 
					className="textField" 
					label="Source" 
					value={inputValue}
					onChange={(event) => setInputValue(event.target.value)}
					multiline
					rows={15} required/>
				<TextField 
					className="textField" 
					label="CT Output" 
					multiline 
					value={outputValue}
					rows={15} />
				<TextField 
					className="textField" 
					label="Output" 
					multiline 
					value={noKnnValue}
					rows={15} />
			</div>
			<TextField 
				className="id" 
				label="Domain Identifier"
				value={domainValue}
				onChange={(event) => setDomainValue(event.target.value)}
				required	
			/>
			<TextField 
				className="id" 
				label="Language Direction"
				value={languageDir}
				onChange={(event) => setlanguageDir(event.target.value)}
				required	
			/>
			<div className="buttons">
				{isLoading ? 
				<span style={{ display: "contents" }}>
				<DefaultButton type="submit" onClick={() => handleSubmit(true)} disabled={isLoading} text="Translating" />
				<DefaultButton type="button" onClick={() => handleSubmit(false)} disabled={isLoading} text="Translating" />
				</span>
				: 
				<span style={{ display: "contents" }}>
				<DefaultButton type="submit" onClick={() => handleSubmit(true)} disabled={!inputValue || !domainValue} text="Custom Translate" />	
				<DefaultButton type="button" onClick={() => handleSubmit(false)} disabled={!inputValue || !domainValue} text="Translate" />
				</span>
				}
			</div>
		</div>
	);
};
