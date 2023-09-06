import "./App.css";
import { Pivot, PivotItem } from "@fluentui/react";
import { createTheme } from "@fluentui/react";
import { Tab1 } from "./Tab1";
import { Tab2 } from "./Tab2";




function App() {
  const myTheme = createTheme({
    palette: {
      themePrimary: '#1f4e78',
      themeLighterAlt: '#f2f6fa',
      themeLighter: '#cedce9',
      themeLight: '#a7c0d6',
      themeTertiary: '#618aae',
      themeSecondary: '#2f5f88',
      themeDarkAlt: '#1c476c',
      themeDark: '#183c5b',
      themeDarker: '#112c43',
      neutralLighterAlt: '#faf9f8',
      neutralLighter: '#f3f2f1',
      neutralLight: '#edebe9',
      neutralQuaternaryAlt: '#e1dfdd',
      neutralQuaternary: '#d0d0d0',
      neutralTertiaryAlt: '#c8c6c4',
      neutralTertiary: '#a19f9d',
      neutralSecondary: '#605e5c',
      neutralSecondaryAlt: '#8a8886',
      neutralPrimaryAlt: '#3b3a39',
      neutralPrimary: '#323130',
      neutralDark: '#201f1e',
      black: '#000000',
      white: '#ffffff',
    }});
	return (
    <div>
		<div className="logo-container">
        {/* <img src="/path/to/logo.png" alt="Logo" />*/}
        <h3>ReAdapt: on-Demand Efficient Custom Translation</h3>
        <p>Get tailored, cost-effective translations without sharing your data! Our machine translation adapts to fit your needs and is resistant to noise for top-notch results.</p>
      </div>
      <div className="App">
      
			<Pivot theme={myTheme} className="tabs">
				<PivotItem key="tab1" headerText="Translation" itemKey="Tab 1">
					<Tab1 />
				</PivotItem>
				<PivotItem key="tab2" headerText="Create Domain" itemKey="Tab 2">
					<Tab2 />
				</PivotItem>
			</Pivot>
		</div>
    </div>
	);
}

export default App;
